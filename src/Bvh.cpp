#include "Bvh.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dependencies/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>

using namespace BvhConstruction;

//For debug purpose
void TraversalCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height)
{

	for (int gIdx = 0; gIdx < width; gIdx++)
	{
		for (int gIdy = 0; gIdy < height; gIdy++)
		{
			u32 nodeIdx = 0;
			u32 top = 0;
			u32 stack[64];
			stack[top++] = INVALID_NODE_IDX;
			HitInfo hit;
			u32 index = gIdx * width + gIdy;

			Ray ray = rayBuff[index];
			Ray transformedRay;
			transformedRay.m_origin = invTransform(ray.m_origin, t.m_scale, t.m_quat, t.m_translation);
			transformedRay.m_direction = invTransform(ray.m_direction, t.m_scale, t.m_quat, { 0.0f,0.0f,0.0f });
			float3 invRayDir = 1.0f / transformedRay.m_direction;

			while (nodeIdx != INVALID_NODE_IDX)
			{
				const LbvhNode& node = bvhNodes[nodeIdx];

				if (LbvhNode::isLeafNode(node))
				{
					Triangle& triangle = primitives[node.m_primIdx];
					float3 tV0 = transform(triangle.v1, t.m_scale, t.m_quat, t.m_translation);
					float3 tV1 = transform(triangle.v2, t.m_scale, t.m_quat, t.m_translation);
					float3 tV2 = transform(triangle.v3, t.m_scale, t.m_quat, t.m_translation);

					float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);
					if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
					{
						hit.m_primIdx = node.m_primIdx;
						hit.m_t = itr.w;
						hit.m_uv = { itr.x, itr.y };
					}
				}
				else
				{
					const Aabb left = bvhNodes[node.m_leftChildIdx].m_aabb;
					const Aabb right = bvhNodes[node.m_rightChildIdx].m_aabb;
					const float2 t0 = left.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const float2 t1 = right.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const bool hitLeft = (t0.x <= t0.y);
					const bool hitRight = (t1.x <= t1.y);

					if (hitLeft || hitRight)
					{
						if (hitLeft && hitRight)
						{
							nodeIdx = (t0.x < t1.x) ? node.m_leftChildIdx : node.m_rightChildIdx;
							if (top < 64)
							{
								stack[top++] = (t0.x < t1.x) ? node.m_rightChildIdx : node.m_leftChildIdx;
							}
						}
						else
						{
							nodeIdx = (hitLeft) ? node.m_leftChildIdx : node.m_rightChildIdx;
						}
						continue;
					}
				}
				nodeIdx = stack[--top];
			}

			if (hit.m_primIdx != INVALID_PRIM_IDX)
			{
				dst[index * 4 + 0] = (hit.m_t / 30.0f) * 255;
				dst[index * 4 + 1] = (hit.m_t / 30.0f) * 255;
				dst[index * 4 + 2] = (hit.m_t / 30.0f) * 255;
				dst[index * 4 + 3] = 255;
			}
		}
	}
}

void BvhConstruction::LBVH::build(Context& context, std::vector<Triangle>& primitives)
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
			"../src/LbvhKernel.h",
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
			"../src/LbvhKernel.h",
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
	d_bvhNodes.resize(nInternalNodes + nLeafNodes);
	{
		{
			Kernel initBvhNodesKernel;

			buildKernelFromSrc(
				initBvhNodesKernel,
				context.m_orochiDevice,
				"../src/LbvhKernel.h",
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
				"../src/LbvhKernel.h",
				"BvhBuild",
				std::nullopt);

			bvhBuildKernel.setArgs({ d_bvhNodes.ptr(), d_sortedMortonCodeKeys.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { bvhBuildKernel.launch(nInternalNodes); });
		}

#if _DEBUG
		const auto debugBuiltNodes = d_bvhNodes.getData();
#endif

		d_flags.resize(nInternalNodes); d_flags.reset();
		{
			Kernel fitBvhNodesKernel;

			buildKernelFromSrc(
				fitBvhNodesKernel,
				context.m_orochiDevice,
				"../src/LbvhKernel.h",
				"FitBvhNodes",
				std::nullopt);

			fitBvhNodesKernel.setArgs({ d_bvhNodes.ptr(), d_flags.ptr(), nLeafNodes, nInternalNodes });
			m_timer.measure(TimerCodes::BvhBuildTime, [&]() { fitBvhNodesKernel.launch(nLeafNodes); });
		}
	}

#if _DEBUG
	const auto debugFittedNodes = d_bvhNodes.getData();
#endif
}

void LBVH::traverseBvh(Context& context)
{
	//set transformation for the scene (fixed currently for cornell box)
	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
	t.m_quat = qtGetIdentity();
	Oro::GpuMemory<Transformation> d_transformations(1); d_transformations.reset();
	OrochiUtils::copyHtoD(d_transformations.ptr(), &t, 1);

	//create camera 
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
			"../src/LbvhKernel.h",
			"GenerateRays",
			std::nullopt);

		generateRaysKernel.setArgs({ d_cam.ptr(), d_rayBuffer.ptr(), width, height });
		m_timer.measure(TimerCodes::BvhBuildTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

	//#if _DEBUG
	//		const auto debugRayBuff = d_rayBuffer.getData();
	//		const auto debugBvhNodes = d_bvhNodes.getData();
	//#endif


#if _CPU																																																						
		//CPU traversal 
	const u32 launchSize = width * height;
	std::vector<HitInfo> h_hitInfo;
	u8* colorBuffer = (u8*)malloc(launchSize * 4);
	memset(dst, 0, launchSize * 4);

	TraversalCPU(debugRayBuff, debugBvhNodes, debugTriangle, t, colorBuffer, width, height);

	stbi_write_png("test.png", width, height, 4, colorBuffer, width * 4);
	free(colorBuffer);
#else

		Oro::GpuMemory<u8> d_colorBuffer(width* height * 4); d_colorBuffer.reset();
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
				"../src/LbvhKernel.h",
				"BvhTraversal",
				std::nullopt);

			traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), width, height });
			m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
		}

		stbi_write_png("test.png", width, height, 4, d_colorBuffer.getData().data(), width * 4);
#endif 

	std::cout << "==========================Perf Times==========================" << std::endl;
	std::cout << "CalculateCentroidExtentsTime :" << m_timer.getTimeRecord(CalculateCentroidExtentsTime) << "ms" << std::endl;
	std::cout << "CalculateMortonCodesTime :" << m_timer.getTimeRecord(CalculateMortonCodesTime) << "ms" << std::endl;
	std::cout << "SortingTime : " << m_timer.getTimeRecord(SortingTime) << "ms" << std::endl;
	std::cout << "BvhBuildTime : " << m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "TraversalTime : " << m_timer.getTimeRecord(TraversalTime) << "ms" << std::endl;
	std::cout << "Total Time : " << m_timer.getTimeRecord(CalculateCentroidExtentsTime) + m_timer.getTimeRecord(CalculateMortonCodesTime) +
		m_timer.getTimeRecord(SortingTime) + m_timer.getTimeRecord(BvhBuildTime) + m_timer.getTimeRecord(TraversalTime) << "ms" << std::endl;
	std::cout << "==============================================================" << std::endl;
}

void SahBvh::build(Context& context, std::vector<Triangle>& primitives)
{
	struct PrimitveRef
	{
		Aabb m_aabb; //world space aabb
		size_t m_primId;
	};

	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
	t.m_quat = qtGetIdentity();

	std::vector<PrimitveRef> primRefs;

	for (size_t i = 0; i < primitives.size(); i++)
	{
		Aabb aabb;
		Triangle& triangle = primitives[i];
		float3 tV0 = transform(triangle.v1, t.m_scale, t.m_quat, t.m_translation);
		float3 tV1 = transform(triangle.v2, t.m_scale, t.m_quat, t.m_translation);
		float3 tV2 = transform(triangle.v3, t.m_scale, t.m_quat, t.m_translation);

		aabb.grow(tV0); aabb.grow(tV1); aabb.grow(tV2);
		primRefs.push_back({ aabb, i });
	}

	struct Buckets
	{
		int m_nPrims = 0;
		Aabb m_aabb;
	};
	constexpr u32 nBuckets = 12;

	u32 primCount = primitives.size();
	m_bvhNodes.resize((2 * primCount - 1) + primCount);

	std::queue<Task> taskQueue;
	Task root = {0,  0, primitives.size() };
	taskQueue.push(root);

	m_bvhNodes.emplace_back();
	SahBvhNode& rootNode = m_bvhNodes[m_bvhNodes.size()];
	rootNode.m_firstChildIdx = 0;
	rootNode.m_primCount = 0;

	while (!taskQueue.empty())
	{
		Task t = taskQueue.back(); taskQueue.pop();
		SahBvhNode& node = m_bvhNodes[t.m_nodeIdx];

		if (t.m_end - t.m_start == 1)
		{

			node.m_firstChildIdx = t.m_start;
			node.m_primCount = t.m_end = t.m_start;
			continue;
		}

		Aabb nodeAabb;
		for (size_t i = t.m_start; i < t.m_end; i++)
		{
			nodeAabb.grow(primRefs[i].m_aabb);
		}
		node.m_aabb = nodeAabb;
		int dim = nodeAabb.maximumExtentDim();

		m_bvhNodes.emplace_back();
		node.m_firstChildIdx = m_bvhNodes.size();
		node.m_primCount = 0;

		SahBvhNode& leftNode = m_bvhNodes[node.m_firstChildIdx];
		m_bvhNodes.emplace_back();
		SahBvhNode& rightNode = m_bvhNodes[node.m_firstChildIdx + 1];

		Buckets buckets[nBuckets];
		for (size_t i = t.m_start; i < t.m_end; i++)
		{
			float centroidDim = 0.0f;
			if(dim == 0) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).x;
			if(dim == 1) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).y;
			if(dim == 2) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).z; 
			u32 b = nBuckets * centroidDim;

			buckets[b].m_nPrims++;
			buckets[b].m_aabb.grow(primRefs[i].m_aabb);
		}

		float cost[nBuckets];
		for (size_t b = 0; b < nBuckets; b++)
		{
			Aabb leftHalf, rightHalf;
			int leftPrimsCount = 0, rightPrimsCount = 0;

			for (size_t j = 0; j <= b; j++)
			{
				leftHalf.grow(buckets[j].m_aabb);
				leftPrimsCount += buckets[j].m_nPrims;
			}

			for (size_t j = b + 1; j < nBuckets; j++)
			{
				rightHalf.grow(buckets[j].m_aabb);
				rightPrimsCount += buckets[j].m_nPrims;
			}

			cost[b] = 0.125f + ((leftPrimsCount * leftHalf.area() + rightPrimsCount * rightHalf.area()) / nodeAabb.area());
		}

		float minCost = cost[0];
		int splitBucket = 0;

		for (size_t i = 0; i < nBuckets - 1; i++)
		{
			if (cost[i] < minCost) {
				minCost = cost[i];
				splitBucket = i;
			}
		}

		u32 split = buckets[splitBucket].m_nPrims - 1;

		Task leftTask = {node.m_firstChildIdx, t.m_start, split };
		Task rightTask = { node.m_firstChildIdx + 1, split + 1, t.m_end };

		taskQueue.push(leftTask);
		taskQueue.push(rightTask);
	}
}

void SahBvh::traverseBvh(Context& context)
{
}
