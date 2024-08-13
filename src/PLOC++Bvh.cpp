#include "PLOC++Bvh.h"
#include <src/Utility.h>
#include <dependencies/stbi/stbi_image_write.h>
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>
#include <assert.h>

//#define WHILEWHILE 1
#define IFIF 1
#define _CPU 1
using namespace BvhConstruction;

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
	assert(Utility::checkPlocBvhCorrectness(h_bvhNodes.data(), h_leafNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
}

void PLOCNew::traverseBvh(Context& context)
{
}
