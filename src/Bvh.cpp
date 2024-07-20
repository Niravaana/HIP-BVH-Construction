#include "Bvh.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dependencies/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>

using namespace BvhConstruction;

/*
*  Bunny 
	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 3.0f, 3.0f, 3.0f };
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

* Sponza 
	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
	t.m_quat = qtRotation(float4{ 1.0f, 0.0f, 0.0f, 1.57f });


	Camera cam;
	cam.m_eye = float4{ -20.0f, 18.5f, 10.8f, 0.0f };
	cam.m_quat = qtRotation(float4{ 0.0f, 1.0f, 0.0f, -1.57f });
	cam.m_fov = 45.0f * Pi / 180.f;
	cam.m_near = 0.0f;
	cam.m_far = 100000.0f;
*/

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

void TraversalSahBvhCPU(const std::vector<Ray>& rayBuff, std::vector<SahBvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height)
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
				const SahBvhNode& node = bvhNodes[nodeIdx];

				if (SahBvhNode::isLeafNode(node))
				{
					Triangle& triangle = primitives[node.m_firstChildIdx];
					float3 tV0 = transform(triangle.v1, t.m_scale, t.m_quat, t.m_translation);
					float3 tV1 = transform(triangle.v2, t.m_scale, t.m_quat, t.m_translation);
					float3 tV2 = transform(triangle.v3, t.m_scale, t.m_quat, t.m_translation);

					float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);
					if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
					{
						hit.m_primIdx = node.m_firstChildIdx;
						hit.m_t = itr.w;
						hit.m_uv = { itr.x, itr.y };
					}
				}
				else
				{
					const Aabb left = bvhNodes[node.m_firstChildIdx].m_aabb;
					const Aabb right = bvhNodes[node.m_firstChildIdx + 1].m_aabb;
					const float2 t0 = left.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const float2 t1 = right.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const bool hitLeft = (t0.x <= t0.y);
					const bool hitRight = (t1.x <= t1.y);

					if (hitLeft || hitRight)
					{
						if (hitLeft && hitRight)
						{
							nodeIdx = (t0.x < t1.x) ? node.m_firstChildIdx : node.m_firstChildIdx + 1;
							if (top < 64)
							{
								stack[top++] = (t0.x < t1.x) ? node.m_firstChildIdx + 1 : node.m_firstChildIdx;
							}
						}
						else
						{
							nodeIdx = (hitLeft) ? node.m_firstChildIdx : node.m_firstChildIdx + 1;
						}
						continue;
					}
				}
				nodeIdx = stack[--top];
			}

			if (hit.m_primIdx != INVALID_PRIM_IDX)
			{
				dst[index * 4 + 0] = (hit.m_uv.x) * 255;
				dst[index * 4 + 1] = (hit.m_uv.y) * 255;
				dst[index * 4 + 2] = (1 - hit.m_uv.x - hit.m_uv.y) * 255;
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
	t.m_scale = float3{ 3.0f, 3.0f, 3.0f };
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
	memset(colorBuffer, 0, launchSize * 4);

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
		m_timer.getTimeRecord(SortingTime) + m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "==============================================================" << std::endl;
}

void SahBvh::build(Context& context, std::vector<Triangle>& primitives)
{
	std::vector<PrimitveRef> primRefs;

	for (size_t i = 0; i < primitives.size(); i++)
	{
		Aabb aabb;
		Triangle& triangle = primitives[i];
		aabb.grow(triangle.v1); aabb.grow(triangle.v2); aabb.grow(triangle.v3);
		primRefs.push_back({ aabb, i });
	}

	struct Buckets
	{
		int m_nPrims = 0;
		Aabb m_aabb;
	};
	constexpr u32 nBuckets = 12;

	u32 primCount = primitives.size();

	std::queue<Task> taskQueue;
	Task root = {0,  0, primitives.size() };
	taskQueue.push(root);

	u32 bvhNodeIdx = 0;
	m_bvhNodes.resize((2 * primCount - 1) + primCount);

	SahBvhNode& rootNode = m_bvhNodes[bvhNodeIdx++];
	rootNode.m_firstChildIdx = 0;
	rootNode.m_primCount = 0;

	while (!taskQueue.empty())
	{
		Task t = taskQueue.front(); taskQueue.pop();
		SahBvhNode& node = m_bvhNodes[t.m_nodeIdx];

		if (t.m_end - t.m_start == 1)
		{
			node.m_aabb = primRefs[t.m_start].m_aabb;
			node.m_firstChildIdx = primRefs[t.m_start].m_primId;
			node.m_primCount = t.m_end - t.m_start;
			continue;
		}

		Aabb nodeAabb;
		for (size_t i = t.m_start; i < t.m_end; i++)
		{
			nodeAabb.grow(primRefs[i].m_aabb);
		}
		node.m_aabb = nodeAabb;
		int dim = nodeAabb.maximumExtentDim();

		node.m_firstChildIdx = bvhNodeIdx++;
		node.m_primCount = 0;
		bvhNodeIdx++;// reserve for right child

		u32 split = 0;

		if (t.m_end - t.m_start <= 2)
		{
			split = (t.m_start + t.m_end) / 2;
			std::nth_element(&primRefs[t.m_start], &primRefs[split],
				&primRefs[t.m_end - 1],
				[dim](const PrimitveRef& a,
					const PrimitveRef& b) {

						float centroidDimA = 0.0f; float centroidDimB = 0.0f;
						if (dim == 0) centroidDimA = a.m_aabb.center().x;
						if (dim == 1) centroidDimA = a.m_aabb.center().y;
						if (dim == 2) centroidDimA = a.m_aabb.center().z;

						if (dim == 0) centroidDimB = b.m_aabb.center().x;
						if (dim == 1) centroidDimB = b.m_aabb.center().y;
						if (dim == 2) centroidDimB = b.m_aabb.center().z;

						return centroidDimA < centroidDimB;
				});
		}
		else
		{
			Buckets buckets[nBuckets];
			for (size_t i = t.m_start; i < t.m_end; i++)
			{
				float centroidDim = 0.0f;
				if (dim == 0) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).x;
				if (dim == 1) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).y;
				if (dim == 2) centroidDim = nodeAabb.offset(primRefs[i].m_aabb.center()).z;
				u32 b = nBuckets * centroidDim;

				buckets[b].m_nPrims++;
				buckets[b].m_aabb.grow(primRefs[i].m_aabb);
			}

			std::vector<float> cost(nBuckets, FltMax);
			for (size_t b = 0; b < nBuckets; b++)
			{
				Aabb leftHalf, rightHalf;
				int leftPrimsCount = 0, rightPrimsCount = 0;

				for (size_t j = 0; j <= b; j++)
				{
					if (buckets[j].m_nPrims == 0) continue;
					leftHalf.grow(buckets[j].m_aabb);
					leftPrimsCount += buckets[j].m_nPrims;
				}

				for (size_t j = b + 1; j < nBuckets; j++)
				{
					if (buckets[j].m_nPrims == 0) continue;
					rightHalf.grow(buckets[j].m_aabb);
					rightPrimsCount += buckets[j].m_nPrims;
				}

				float leftSahCost = (leftPrimsCount == 0) ? 0.0f : leftPrimsCount * leftHalf.area();
				float rightSahCost = (rightPrimsCount == 0) ? 0.0f : rightPrimsCount * rightHalf.area();
				float totalSahCost = (leftPrimsCount + rightPrimsCount) == 0 ? 0.0f : ((leftSahCost + rightSahCost) / nodeAabb.area());
				cost[b] = (totalSahCost == 0.0f) ? FltMax : 0.125f + totalSahCost;
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

			u32 count = 0;
			 split = std::partition(&primRefs[t.m_start], &primRefs[t.m_end - 1], [&](const PrimitveRef& pi) {
				float centroidDim = 0.0f;
				if (dim == 0) centroidDim = nodeAabb.offset(pi.m_aabb.center()).x;
				if (dim == 1) centroidDim = nodeAabb.offset(pi.m_aabb.center()).y;
				if (dim == 2) centroidDim = nodeAabb.offset(pi.m_aabb.center()).z;
				u32 b = nBuckets * centroidDim;
				if (b == nBuckets) b = nBuckets - 1; count++;
				return b <= splitBucket;
				}) - &primRefs[0];

			 //bucketing failed so lets partition sorting based on nodes centroid
			 if (split <= t.m_start || split >= t.m_end)
			 {
				 float centroidDim = 0.0f;
				 if (dim == 0) centroidDim = nodeAabb.offset(nodeAabb.center()).x;
				 if (dim == 1) centroidDim = nodeAabb.offset(nodeAabb.center()).y;
				 if (dim == 2) centroidDim = nodeAabb.offset(nodeAabb.center()).z;

				 float mid = centroidDim / 2.0f;

				 split = std::partition(
					 &primRefs[t.m_start], &primRefs[t.m_end - 1],
					 [dim, mid](const PrimitveRef& pi) {
						 float centroidDim = 0.0f;
						 if (dim == 0) centroidDim = pi.m_aabb.center().x;
						 if (dim == 1) centroidDim = pi.m_aabb.center().y;
						 if (dim == 2) centroidDim = pi.m_aabb.center().z;
						 return centroidDim < mid; }) - &primRefs[0];
			 }

			 //If centroid partition also failed 
			 if (split <= t.m_start || split >= t.m_end)
			 {
				 split = (t.m_start + t.m_end) / 2;
				 std::nth_element(&primRefs[t.m_start], &primRefs[split],
					 &primRefs[t.m_end - 1],
					 [dim](const PrimitveRef& a,
						 const PrimitveRef& b) {

							 float centroidDimA = 0.0f; float centroidDimB = 0.0f;
							 if (dim == 0) centroidDimA = a.m_aabb.center().x;
							 if (dim == 1) centroidDimA = a.m_aabb.center().y;
							 if (dim == 2) centroidDimA = a.m_aabb.center().z;

							 if (dim == 0) centroidDimB = b.m_aabb.center().x;
							 if (dim == 1) centroidDimB = b.m_aabb.center().y;
							 if (dim == 2) centroidDimB = b.m_aabb.center().z;

							 return centroidDimA < centroidDimB;
					 });
			 }
		}

		Task leftTask = {node.m_firstChildIdx, t.m_start, split };
		Task rightTask = { node.m_firstChildIdx + 1, split , t.m_end};

		taskQueue.push(leftTask);
		taskQueue.push(rightTask);
	}

// Lets to DFS and collect all primIds should match total primIdx
#if _DEBUG
	u32 stack[32];
	int top = 0;
	stack[top++] = INVALID_NODE_IDX;
	u32 nodeIdx = 0;
	std::vector<u32> primIdxs;
	
	while (nodeIdx != INVALID_NODE_IDX)
	{
		SahBvhNode node = m_bvhNodes[nodeIdx];
		if (SahBvhNode::isLeafNode(node))
		{
			primIdxs.push_back(node.m_firstChildIdx);
		}
		else
		{
			stack[top++] = node.m_firstChildIdx;
			stack[top++] = node.m_firstChildIdx + 1;
		}
		nodeIdx = stack[--top];
	}

#endif 

	Transformation t;
	t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
	t.m_scale = float3{ 3.0f, 3.0f, 3.0f };
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
	Oro::GpuMemory<Ray> d_rayBuffer(width* height); d_rayBuffer.reset();

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
		generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1);
	}

	const auto debugRayBuff = d_rayBuffer.getData();

	//CPU traversal 
	const u32 launchSize = width * height;
	std::vector<HitInfo> h_hitInfo;
	u8* colorBuffer = (u8*)malloc(launchSize * 4);
	memset(colorBuffer, 0, launchSize * 4);

	TraversalSahBvhCPU(debugRayBuff, m_bvhNodes, primitives, t, colorBuffer, width, height);

	stbi_write_png("test.png", width, height, 4, colorBuffer, width * 4);

	std::cout << "Done!!";
}

void SahBvh::traverseBvh(Context& context)
{
	
}
