#include "PLOC++Bvh.h"
#include <src/Utility.h>
#include <dependencies/stbi/stbi_image_write.h>
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>
#include <assert.h>

using namespace BvhConstruction;

//#define WHILEWHILE 1
#define IFIF 1
#define USE_GPU_WIDE_COLLAPSE 1

/*
ToDo : 
1. For primNode we might not need parentIdx 
*/
void PLOCNew::build(Context& context, std::vector<Triangle>& primitives)
{
 	const size_t primitiveCount = primitives.size();
	d_triangleBuff.resize(primitiveCount); d_triangleBuff.reset();
	d_triangleAabb.resize(primitiveCount); d_triangleAabb.reset(); //ToDo we might not need it.
	OrochiUtils::copyHtoD(d_triangleBuff.ptr(), primitives.data(), primitives.size());

	d_sceneExtents.resize(1); d_sceneExtents.reset();
	Aabb extent; extent.reset();
	OrochiUtils::copyHtoD(d_sceneExtents.ptr(), &extent, 1);
	{
		Kernel centroidExtentsKernel;

		buildKernelFromSrc(
			centroidExtentsKernel,
			context.m_orochiDevice,
			"../src/CommonBlocksKernel.h",
			"CalculateSceneExtents",
			std::nullopt);

		centroidExtentsKernel.setArgs({ d_triangleBuff.ptr(), d_triangleAabb.ptr(), d_sceneExtents.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateCentroidExtentsTime, [&]() { centroidExtentsKernel.launch(primitiveCount, ReductionBlockSize); });
	}

	d_mortonCodeKeys.resize(primitiveCount); d_mortonCodeKeys.reset();
	d_mortonCodeValues.resize(primitiveCount); d_mortonCodeValues.reset();
	d_sortedMortonCodeKeys.resize(primitiveCount); d_sortedMortonCodeKeys.reset();
	d_sortedMortonCodeValues.resize(primitiveCount); d_sortedMortonCodeValues.reset();
	{
		Kernel calulateMortonCodesKernel;

		buildKernelFromSrc(
			calulateMortonCodesKernel,
			context.m_orochiDevice,
			"../src/CommonBlocksKernel.h",
			"CalculateMortonCodes",
			std::nullopt);

		calulateMortonCodesKernel.setArgs({ d_triangleAabb.ptr(), d_sceneExtents.ptr() , d_mortonCodeKeys.ptr(), d_mortonCodeValues.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateMortonCodesTime, [&]() { calulateMortonCodesKernel.launch(primitiveCount); });
	}

#if _DEBUG
	const auto debugMortonCodes = d_mortonCodeKeys.getData();
#endif

	{
		OrochiUtils oroUtils;
		Oro::RadixSort sort(context.m_orochiDevice, oroUtils, 0, "../dependencies/Orochi/ParallelPrimitives/RadixSortKernels.h", "../dependencies/Orochi");

		Oro::RadixSort::KeyValueSoA srcGpu{};
		Oro::RadixSort::KeyValueSoA dstGpu{};
		static constexpr auto startBit{ 0 };
		static constexpr auto endBit{ 32 };
		static constexpr auto stream = 0;

		srcGpu.key = d_mortonCodeKeys.ptr();
		srcGpu.value = d_mortonCodeValues.ptr();

		dstGpu.key = d_sortedMortonCodeKeys.ptr();
		dstGpu.value = d_sortedMortonCodeValues.ptr();

		m_timer.measure(SortingTime, [&]() {
			sort.sort(srcGpu, dstGpu, static_cast<int>(primitiveCount), startBit, endBit, stream); });
	}

	const u32 nLeafNodes = primitiveCount;
	const u32 nInternalNodes = nLeafNodes - 1;
	bool swapBuffer = false;
	u32 nClusters = primitiveCount;

	d_leafNodes.resize(primitiveCount); d_leafNodes.reset();
	d_bvhNodes.resize(nInternalNodes); d_bvhNodes.reset();

	int invalid = INVALID_NODE_IDX;
	Oro::GpuMemory<int> d_nodeIdx0(primitiveCount); 
	Oro::GpuMemory<int> d_nodeIdx1(primitiveCount); 
	Oro::GpuMemory<int> d_nMergedClusters(1); 
	Oro::GpuMemory<int> d_blockOffsetSum(1);
	Oro::GpuMemory<int> d_atomicBlockCounter(1);
	
	OrochiUtils::memset(d_nodeIdx0.ptr(), invalid, sizeof(int) * primitiveCount);
	OrochiUtils::memset(d_nodeIdx1.ptr(), invalid, sizeof(int) * primitiveCount);
	
	{
		Kernel setupClusterKernel;

		buildKernelFromSrc(
			setupClusterKernel,
			context.m_orochiDevice,
			"../src/Ploc++Kernel.h",
			"SetupClusters",
			std::nullopt);

		setupClusterKernel.setArgs({d_bvhNodes.ptr(),  d_leafNodes.ptr(), d_sortedMortonCodeValues.ptr() , d_triangleAabb.ptr(), d_nodeIdx0.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateMortonCodesTime, [&]() { setupClusterKernel.launch(primitiveCount); });
	}

	Kernel plocKernel;

	buildKernelFromSrc(
		plocKernel,
		context.m_orochiDevice,
		"../src/Ploc++Kernel.h",
		"Ploc",
		std::nullopt);

	while (nClusters > 1)
	{
		d_nMergedClusters.reset(); d_blockOffsetSum.reset(); d_atomicBlockCounter.reset();
		int* nodeIndices0 = !swapBuffer ? d_nodeIdx0.ptr() : d_nodeIdx1.ptr();
		int* nodeIndices1 = !swapBuffer ? d_nodeIdx1.ptr() : d_nodeIdx0.ptr();

		{
			plocKernel.setArgs({ nodeIndices0, nodeIndices1, d_bvhNodes.ptr(), d_leafNodes.ptr(), d_nMergedClusters.ptr(), d_blockOffsetSum.ptr(),  d_atomicBlockCounter.ptr(), nClusters, nInternalNodes});
			m_timer.measure(TimerCodes::CalculateMortonCodesTime, [&]() { plocKernel.launch(nClusters, PlocBlockSize); });
		}

		nClusters = nClusters - d_nMergedClusters.getData()[0];
		swapBuffer = !swapBuffer;
	}

	const auto h_bvhNodes = d_bvhNodes.getData();
	const auto h_leafNodes = d_leafNodes.getData();
	assert(Utility::checkPlocBvh2Correctness(h_bvhNodes.data(), h_leafNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);

	Oro::GpuMemory<Bvh4Node> d_wideBvhNodes(2 * primitiveCount); d_wideBvhNodes.reset();
	Oro::GpuMemory<PrimNode> d_wideLeafNodes(primitiveCount); d_wideLeafNodes.reset();
	uint2 invTask = { INVALID_NODE_IDX, INVALID_NODE_IDX };
	uint2 rootTask = { 0, INVALID_NODE_IDX };
	u32 one = 1;
	u32 zero = 0;
	Oro::GpuMemory<u32> d_taskCounter(1); d_taskCounter.reset();
	Oro::GpuMemory<uint2> d_taskQ(primitiveCount); d_taskQ.reset();
	Oro::GpuMemory<u32> d_internalNodeOffset(1); d_internalNodeOffset.reset();
	std::vector<uint2> invTasks(primitiveCount, invTask);

	OrochiUtils::copyHtoD(d_taskCounter.ptr(), &zero, 1);
	OrochiUtils::copyHtoD(d_taskQ.ptr(), invTasks.data(), invTasks.size());
	OrochiUtils::copyHtoD(d_taskQ.ptr(), &rootTask, 1);
	OrochiUtils::copyHtoD(d_internalNodeOffset.ptr(), &one, 1);

	const auto debugTaskQ = d_taskQ.getData();
	{
		Kernel collapseToWide4BvhKernel;

		buildKernelFromSrc(
			collapseToWide4BvhKernel,
			context.m_orochiDevice,
			"../src/Ploc++Kernel.h",
			"CollapseToWide4Bvh",
			std::nullopt);

		collapseToWide4BvhKernel.setArgs({ d_bvhNodes.ptr(), d_leafNodes.ptr(), d_wideBvhNodes.ptr(), d_wideLeafNodes.ptr(), d_taskQ.ptr(), d_taskCounter.ptr(),  d_internalNodeOffset.ptr(), nInternalNodes, nLeafNodes });
		m_timer.measure(TimerCodes::CollapseBvhTime, [&]() { collapseToWide4BvhKernel.launch(divideRoundUp(2 * primitiveCount, 3)); });
	}

	const auto wideBvhNodes = d_wideBvhNodes.getData();
	const auto wideLeafNodes = d_wideLeafNodes.getData();
	auto internalNodeOffset = d_internalNodeOffset.getData()[0];
	auto triangleAabb = d_triangleAabb.getData();

	assert(Utility::checkLBvh4Correctness(wideBvhNodes.data(), wideLeafNodes.data(), m_rootNodeIdx, nInternalNodes) == true);
	m_cost = Utility::calculatebvh4Cost(wideBvhNodes.data(), wideLeafNodes.data(), triangleAabb.data(), m_rootNodeIdx, internalNodeOffset, nInternalNodes);
}

void PLOCNew::traverseBvh(Context& context)
{
	//Traversal code will need to be changed for Ploc

	std::cout << "==========================Perf Times==========================" << std::endl;
	std::cout << "CalculateCentroidExtentsTime :" << m_timer.getTimeRecord(CalculateCentroidExtentsTime) << "ms" << std::endl;
	std::cout << "CalculateMortonCodesTime :" << m_timer.getTimeRecord(CalculateMortonCodesTime) << "ms" << std::endl;
	std::cout << "SortingTime : " << m_timer.getTimeRecord(SortingTime) << "ms" << std::endl;
	std::cout << "BvhBuildTime : " << m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "TraversalTime : " << m_timer.getTimeRecord(TraversalTime) << "ms" << std::endl;
	std::cout << "CollapseTime : " << m_timer.getTimeRecord(CollapseBvhTime) << "ms" << std::endl;
	std::cout << "Bvh Cost : " << m_cost << std::endl;
	std::cout << "Total Time : " << m_timer.getTimeRecord(CalculateCentroidExtentsTime) + m_timer.getTimeRecord(CalculateMortonCodesTime) +
		m_timer.getTimeRecord(SortingTime) + m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "==============================================================" << std::endl;
}
