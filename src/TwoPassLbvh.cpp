#include "TwoPassLbvh.h"
#include <src/Utility.h>
#include <dependencies/stbi/stbi_image_write.h>
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>
#include <assert.h>

using namespace BvhConstruction;

#define WHILEWHILE 1
//#define IFIF 1
//#define USE_PRIM_SPLITTING 1
#define USE_GPU_WIDE_COLLAPSE 1

void TwoPassLbvh::build(Context& context, std::vector<Triangle>& primitives)
{
	d_triangleBuff.resize(primitives.size()); d_triangleBuff.reset();
	OrochiUtils::copyHtoD(d_triangleBuff.ptr(), primitives.data(), primitives.size());

	std::vector<PrimRef> h_primRefs;
#ifdef USE_PRIM_SPLITTING
	float saMax = 10.0f;
	Utility::doEarlySplitClipping(primitives, h_primRefs, saMax);
#else
	Utility::doEarlySplitClipping(primitives, h_primRefs);
#endif

	const u32 primitiveCount = h_primRefs.size();
	Oro::GpuMemory<PrimRef> d_primRefs(primitiveCount); d_primRefs.reset();
	OrochiUtils::copyHtoD(d_primRefs.ptr(), h_primRefs.data(), primitiveCount);

	d_sceneExtents.resize(1); d_sceneExtents.reset();
	Aabb emptyExtents; emptyExtents.reset();
	OrochiUtils::copyHtoD(d_sceneExtents.ptr(), &emptyExtents, 1);
	{
		Kernel centroidExtentsKernel;

		buildKernelFromSrc(
			centroidExtentsKernel,
			context.m_orochiDevice,
			"../src/CommonBlocksKernel.h",
			"CalculatePrimRefExtents",
			std::nullopt);

		centroidExtentsKernel.setArgs({ d_primRefs.ptr() , d_sceneExtents.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateCentroidExtentsTime, [&]() { centroidExtentsKernel.launch(primitiveCount); });
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
			"CalculateMortonCodesPrimRef",
			std::nullopt);

		calulateMortonCodesKernel.setArgs({ d_primRefs.ptr(), d_sceneExtents.ptr() , d_mortonCodeKeys.ptr(), d_mortonCodeValues.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateMortonCodesTime, [&]() { calulateMortonCodesKernel.launch(primitiveCount); });
	}

	const auto debugPrimRefs = d_primRefs.getData();
	const auto ext = d_sceneExtents.getData();
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
	const u32 nTotalNodes = nInternalNodes + nLeafNodes;
	m_nInternalNodes = nInternalNodes;
	d_bvhNodes.resize(nTotalNodes);
	Oro::GpuMemory<u32> d_parentIdxs(nTotalNodes);
	{
		{
			Kernel initBvhNodesKernel;

			buildKernelFromSrc(
				initBvhNodesKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"InitBvhNodesPrimRef",
				std::nullopt);

			initBvhNodesKernel.setArgs({ d_primRefs.ptr(), d_bvhNodes.ptr(), d_parentIdxs.ptr(), d_sortedMortonCodeValues.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { initBvhNodesKernel.launch(nLeafNodes); });
		}

#if _DEBUG
		const auto debugNodes = d_bvhNodes.getData();
#endif

		{
			Kernel bvhBuildKernel;

			buildKernelFromSrc(
				bvhBuildKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"BvhBuild",
				std::nullopt);

			bvhBuildKernel.setArgs({ d_bvhNodes.ptr(), d_parentIdxs.ptr(), d_sortedMortonCodeKeys.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { bvhBuildKernel.launch(nInternalNodes); });
		}

		d_flags.resize(nTotalNodes); d_flags.reset();
		{
			Kernel fitBvhNodesKernel;

			buildKernelFromSrc(
				fitBvhNodesKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"FitBvhNodes",
				std::nullopt);

			fitBvhNodesKernel.setArgs({ d_bvhNodes.ptr(), d_parentIdxs.ptr(), d_flags.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { fitBvhNodesKernel.launch(nLeafNodes); });
		}
	}
		const auto h_bvhNodes = d_bvhNodes.getData();

#if _DEBUG
		
		assert(Utility::checkLbvhRootAabb(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		assert(Utility::checkLBvhCorrectness(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		m_cost = Utility::calculateLbvhCost(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes);
#endif

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
				"../src/TwoPassLbvhKernel.h",
				"CollapseToWide4Bvh",
				std::nullopt);

			collapseToWide4BvhKernel.setArgs({ d_bvhNodes.ptr(), d_wideBvhNodes.ptr(), d_wideLeafNodes.ptr(), d_taskQ.ptr(), d_taskCounter.ptr(),  d_internalNodeOffset.ptr(), nInternalNodes, nLeafNodes });
			m_timer.measure(TimerCodes::CollapseBvhTime, [&]() { collapseToWide4BvhKernel.launch(divideRoundUp(2 * primitiveCount, 3)); });
		}

		const auto wideBvhNodes = d_wideBvhNodes.getData();
		const auto wideLeafNodes = d_wideLeafNodes.getData();
		auto internalNodeOffset = d_internalNodeOffset.getData()[0];
		auto primRefs = d_primRefs.getData();
		std::vector<Aabb> triangleAabb(primRefs.size());
		for (size_t i = 0; i < primRefs.size(); i++)
		{
			triangleAabb[primRefs[i].m_primIdx] = primRefs[i].m_aabb;
		}

		assert(Utility::checkLBvh4Correctness(wideBvhNodes.data(), wideLeafNodes.data(), m_rootNodeIdx, nInternalNodes) == true);
		m_cost = Utility::calculatebvh4Cost(wideBvhNodes.data(), wideLeafNodes.data(), triangleAabb.data(), m_rootNodeIdx, internalNodeOffset, nInternalNodes);
}

void TwoPassLbvh::traverseBvh(Context& context)
{

	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -5.0f };
	t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
	t.m_quat = qtGetIdentity();
	Oro::GpuMemory<Transformation> d_transformations(1); d_transformations.reset();
	OrochiUtils::copyHtoD(d_transformations.ptr(), &t, 1);

	Camera cam;

	cam.m_eye = float4{ 0.0f, 2.5f, 5.8f, 0.0f };
	cam.m_quat = qtRotation({ 0.0f, 0.0f, 1.0f, -1.57f });
	cam.m_fov = 45.0f * Pi / 180.f;
	cam.m_near = 0.0f;
	cam.m_far = 100000.0f;

	Oro::GpuMemory<Camera> d_cam(1); d_cam.reset();
	OrochiUtils::copyHtoD(d_cam.ptr(), &cam, 1);


	u32 width = 512;
	u32 height = 512;
	Oro::GpuMemory<Ray> d_rayBuffer(width * height); d_rayBuffer.reset();
	Oro::GpuMemory<u32> d_rayCounterBuffer(width * height); d_rayCounterBuffer.reset();
	//generate rays
	{
		const u32 blockSizeX = 8;
		const u32 blockSizeY = 8;
		const u32 gridSizeX = (width + blockSizeX - 1) / blockSizeX;
		const u32 gridSizeY = (height + blockSizeY - 1) / blockSizeY;
		Kernel generateRaysKernel;

		buildKernelFromSrc(
			generateRaysKernel,
			context.m_orochiDevice,
			"../src/CommonBlocksKernel.h",
			"GenerateRays",
			std::nullopt);

		generateRaysKernel.setArgs({ d_cam.ptr(), d_rayBuffer.ptr(), width, height });
		m_timer.measure(TimerCodes::RayGenTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

	Oro::GpuMemory<u8> d_colorBuffer(width * height * 4); d_colorBuffer.reset();
	/*const auto debugRayBuff = d_rayBuffer.getData();
	const auto debugNodes = d_bvhNodes.getData();
	const auto debugTriBuff = d_triangleBuff.getData();
	auto dst = d_colorBuffer.getData();
	Utility::TraversalLbvhCPU(debugRayBuff, debugNodes, debugTriBuff, t, dst.data(), width, height, m_nInternalNodes);*/
#if defined IFIF

	//Traversal kernel
	{
		const u32 blockSizeX = 8;
		const u32 blockSizeY = 8;
		const u32 gridSizeX = (width + blockSizeX - 1) / blockSizeX;
		const u32 gridSizeY = (height + blockSizeY - 1) / blockSizeY;
		Kernel traversalKernel;

		buildKernelFromSrc(
			traversalKernel,
			context.m_orochiDevice,
			"../src/TraversalKernel.h",
			"BvhTraversalifif",
			std::nullopt);

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_rayCounterBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

#if _DEBUG
	const auto rayCounter = d_rayCounterBuffer.getData();
	Utility::generateTraversalHeatMap(rayCounter, width, height);
#endif 	
#elif defined WHILEWHILE
	OrochiUtils::waitForCompletion();
	//Traversal kernel
	{
		const u32 blockSizeX = 8;
		const u32 blockSizeY = 8;
		const u32 gridSizeX = (width + blockSizeX - 1) / blockSizeX;
		const u32 gridSizeY = (height + blockSizeY - 1) / blockSizeY;
		Kernel traversalKernel;

		buildKernelFromSrc(
			traversalKernel,
			context.m_orochiDevice,
			"../src/TraversalKernel.h",
			"BvhTraversalSpeculativeWhile",
			std::nullopt);

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}
	OrochiUtils::waitForCompletion();
#endif

	stbi_write_png("test.png", width, height, 4, d_colorBuffer.getData().data(), width * 4);

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
