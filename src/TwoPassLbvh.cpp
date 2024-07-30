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
	Oro::GpuMemory<u32> d_parentIdxs(nTotalNodes);
	{
		{
			Kernel initBvhNodesKernel;

			buildKernelFromSrc(
				initBvhNodesKernel,
				context.m_orochiDevice,
				"../src/TwoPassLbvhKernel.h",
				"InitBvhNodes",
				std::nullopt);

			initBvhNodesKernel.setArgs({ d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_parentIdxs.ptr(), d_sortedMortonCodeValues.ptr(), nLeafNodes, nInternalNodes });
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

			fitBvhNodesKernel.setArgs({ d_bvhNodes.ptr(), d_parentIdxs.ptr(), d_flags.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { fitBvhNodesKernel.launch(nLeafNodes); });
		}
		
#if _DEBUG
		const auto h_bvhNodes = d_bvhNodes.getData();
		assert(Utility::checkLbvhRootAabb(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		assert(Utility::checkLBvhCorrectness(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes) == true);
		m_cost = Utility::calculateLbvhCost(h_bvhNodes.data(), m_rootNodeIdx, nLeafNodes, nInternalNodes);
#endif

		std::vector<Bvh4Node> wideBvhNodes(2 * primitiveCount); //will hold internal nodes for wide bvh
		std::vector<PrimNode> wideLeafNodes(primitiveCount); // leaf nodes of wide bvh
		uint2 invTask = { INVALID_NODE_IDX, INVALID_NODE_IDX };
		std::vector<uint2> taskQ(primitiveCount, invTask);
		u32 taskCounter = 0; //when it reached to num of primCounts we break out of loop
		u32 internalNodeOffset = 1; // we will shift it by internal nodes created, set to 1 as root node is the first one
		taskQ[0] = { 0, INVALID_NODE_IDX }; //initially we have only root task 

		/*
		
		  ToDo we no longer need AABB in LBVH node for leaf so we can get rid of it, we can separate primNodes and Bvh2Nodes there too
		
		*/

		for (int i = 0; i < primitiveCount; i++)
		{
			wideLeafNodes[i].m_primIdx = h_bvhNodes[i + nInternalNodes].m_leftChildIdx;
		}

		collapseBvh2toBvh4(h_bvhNodes, wideBvhNodes, wideLeafNodes, taskQ, taskCounter, internalNodeOffset, nInternalNodes, nLeafNodes);

		assert(Utility::checkLBvh4Correctness(wideBvhNodes.data(), wideLeafNodes.data(), m_rootNodeIdx, nInternalNodes) == true);

		std::cout << "Done";
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

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_rayCounterBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
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

void TwoPassLbvh::collapseBvh2toBvh4(const std::vector<LbvhNode>& bvh2Nodes, std::vector<Bvh4Node>& bvh4Nodes, std::vector<PrimNode> bvh4LeafNodes, std::vector<uint2>& taskQ, u32 taskCount, u32 bvh8InternalNodeOffset, u32 nBvh2InternalNodes, u32 nBvh2LeafNodes)
{

	for (u32 index = 0; index < nBvh2LeafNodes; index++)
	{
		if (taskCount == nBvh2LeafNodes) break;
		uint2 task = taskQ[index];
		u32 bvh2NodeIdx = task.x;
		u32 parentIdx = task.y;

		if (bvh2NodeIdx != INVALID_NODE_IDX)
		{
			const LbvhNode& node2 = bvh2Nodes[bvh2NodeIdx];
			u32 childIdx[4] = {INVALID_NODE_IDX, INVALID_NODE_IDX , INVALID_NODE_IDX , INVALID_NODE_IDX };
			Aabb childAabb[4];
			u32 childCount = 2;
			childIdx[0] = node2.m_leftChildIdx;
			childIdx[1] = node2.m_rightChildIdx;
			childAabb[0] = bvh2Nodes[node2.m_leftChildIdx].m_aabb;
			childAabb[1] = bvh2Nodes[node2.m_rightChildIdx].m_aabb;

			for(size_t j = 0; j < 2; j++) //N = 2 so we just need to expand one level to go to grandchildren
			{
				float maxArea = 0.0f;
				u32 maxAreaChildPos = INVALID_NODE_IDX;
				for (size_t k = 0; k < childCount; k++)
				{
					if (childIdx[k] < nBvh2InternalNodes) //this is an intenral node 
					{
						float area = bvh2Nodes[childIdx[k]].m_aabb.area();
						if (area > maxArea)
						{
							maxAreaChildPos = k;
							maxArea = area;
						}
					}
				}

				if (maxAreaChildPos == INVALID_NODE_IDX) break;

				LbvhNode maxChild = bvh2Nodes[childIdx[maxAreaChildPos]];
				childIdx[maxAreaChildPos] = maxChild.m_leftChildIdx;
				childAabb[maxAreaChildPos] = bvh2Nodes[maxChild.m_leftChildIdx].m_aabb;
				childIdx[childCount] = maxChild.m_rightChildIdx;
				childAabb[childCount] = bvh2Nodes[maxChild.m_rightChildIdx].m_aabb;
				childCount++;

			}// while 

			//Here we have all 4 child indices lets create wide node 
			Bvh4Node wideNode;
			wideNode.m_parent = parentIdx;

			for (size_t i = 0; i < childCount; i++)
			{
				if (childIdx[i] < nBvh2InternalNodes)
				{
					wideNode.m_child[i] = bvh8InternalNodeOffset++;
					wideNode.m_aabb[i] = childAabb[i];
					taskQ[wideNode.m_child[i]] = { childIdx[i] , index };
				}
				else
				{
					wideNode.m_child[i] = childIdx[i];
					bvh4LeafNodes[childIdx[i] - nBvh2InternalNodes].m_parent = index;
					taskCount++;
				}
			}

			bvh4Nodes[index] = wideNode;
		}
	}
}