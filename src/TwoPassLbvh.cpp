#include "TwoPassLbvh.h"
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

void TwoPassLbvh::build(Context& context, std::vector<Triangle>& primitives)
{
	const size_t primitiveCount = primitives.size();
	d_triangleBuff.resize(primitiveCount); d_triangleBuff.reset();
	d_triangleAabb.resize(primitiveCount); d_triangleAabb.reset(); //ToDo we might not need it.
	OrochiUtils::copyHtoD(d_triangleBuff.ptr(), primitives.data(), primitives.size());

	d_sceneExtents.resize(1); d_sceneExtents.reset();
	{
		Kernel centroidExtentsKernel;

		buildKernelFromSrc(
			centroidExtentsKernel,
			context.m_orochiDevice,
			"../src/CommonBlocksKernel.h",
			"CalculateSceneExtents",
			std::nullopt);

		centroidExtentsKernel.setArgs({ d_triangleBuff.ptr(), d_triangleAabb.ptr(), d_sceneExtents.ptr(), primitiveCount });
		m_timer.measure(TimerCodes::CalculateCentroidExtentsTime, [&]() { centroidExtentsKernel.launch(primitiveCount); });
	}

#if _DEBUG
	const auto debugTriangle = d_triangleBuff.getData();
	const auto debugAabb = d_triangleAabb.getData();
	const auto debugExtent = d_sceneExtents.getData()[0];
#endif

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

#if _DEBUG
	const auto debugSortedMortonCodes = d_sortedMortonCodeKeys.getData();
	const auto debugSortedMortonCodesVal = d_sortedMortonCodeValues.getData();
#endif

	const u32 nLeafNodes = primitiveCount;
	const u32 nInternalNodes = nLeafNodes - 1;
	const u32 nTotalNodes = nInternalNodes + nLeafNodes;
	m_nInternalNodes = nInternalNodes;
	d_bvhNodes.resize(nTotalNodes);
	{
		{
			Kernel initBvhNodesKernel;

			buildKernelFromSrc(
				initBvhNodesKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"InitBvhNodes",
				std::nullopt);

			initBvhNodesKernel.setArgs({ d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_sortedMortonCodeValues.ptr(), nInternalNodes, nLeafNodes });
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

			bvhBuildKernel.setArgs({ d_bvhNodes.ptr(), d_sortedMortonCodeKeys.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { bvhBuildKernel.launch(nInternalNodes); });
		}

#if _DEBUG
		const auto debugBuiltNodes = d_bvhNodes.getData();
#endif

		d_flags.resize(nTotalNodes); d_flags.reset();
		{
			Kernel fitBvhNodesKernel;

			buildKernelFromSrc(
				fitBvhNodesKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"FitBvhNodes",
				std::nullopt);

			fitBvhNodesKernel.setArgs({ d_bvhNodes.ptr(), d_flags.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { fitBvhNodesKernel.launch(nLeafNodes); });
		}
		
#if _DEBUG
		const auto h_bvhNodes = d_bvhNodes.getData();
		assert(Utility::checkLbvhRootAabb(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		assert(Utility::checkLBvhCorrectness(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		m_cost = Utility::calculateLbvhCost(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes);
#endif
	}
}

void TwoPassLbvh::traverseBvh(Context& context)
{
	//set transformation for the scene (fixed currently for cornell box)
	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 3.0f, 3.0f, 3.0f };
	t.m_quat = qtGetIdentity();
	Oro::GpuMemory<Transformation> d_transformations(1); d_transformations.reset();
	OrochiUtils::copyHtoD(d_transformations.ptr(), &t, 1);

	Camera cam;
	cam.m_eye = float4{ 0.0f, 2.5f, 5.8f, 0.0f };
	cam.m_quat = qtRotation(float4{ 0.0f, 0.0f, 1.0f, -1.57f });
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
		m_timer.measure(TimerCodes::BvhBuildTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

	Oro::GpuMemory<u8> d_colorBuffer(width * height * 4); d_colorBuffer.reset();

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

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_rayCounterBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes * 2 });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

#if _DEBUG
	const auto rayCounter = d_rayCounterBuffer.getData();
	Utility::generateTraversalHeatMap(rayCounter, width, height);
#endif 	
#elif defined WHILEWHILE

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
			"BvhTraversalWhile",
			std::nullopt);

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

#endif

	stbi_write_png("test.png", width, height, 4, d_colorBuffer.getData().data(), width * 4);

	std::cout << "==========================Perf Times==========================" << std::endl;
	std::cout << "CalculateCentroidExtentsTime :" << m_timer.getTimeRecord(CalculateCentroidExtentsTime) << "ms" << std::endl;
	std::cout << "CalculateMortonCodesTime :" << m_timer.getTimeRecord(CalculateMortonCodesTime) << "ms" << std::endl;
	std::cout << "SortingTime : " << m_timer.getTimeRecord(SortingTime) << "ms" << std::endl;
	std::cout << "BvhBuildTime : " << m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "TraversalTime : " << m_timer.getTimeRecord(TraversalTime) << "ms" << std::endl;
	std::cout << "Bvh Cost : " << m_cost << std::endl;
	std::cout << "Total Time : " << m_timer.getTimeRecord(CalculateCentroidExtentsTime) + m_timer.getTimeRecord(CalculateMortonCodesTime) +
		m_timer.getTimeRecord(SortingTime) + m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "==============================================================" << std::endl;
}
