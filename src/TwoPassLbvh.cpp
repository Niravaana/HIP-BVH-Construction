#include "TwoPassLbvh.h"
#include <dependencies/stbi/stbi_image_write.h>
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>

//#define WHILEWHILE 1
#define IFIF 1
#define _CPU 1
using namespace BvhConstruction;

//For debug purpose
static void TraversalCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode32> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height, u32 nInternalNodes)
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
				const LbvhNode32& node = bvhNodes[nodeIdx];

				if (nodeIdx >= nInternalNodes)
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
					const Aabb right = bvhNodes[node.m_leftChildIdx + 1].m_aabb;
					const float2 t0 = left.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const float2 t1 = right.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
					const bool hitLeft = (t0.x <= t0.y);
					const bool hitRight = (t1.x <= t1.y);

					if (hitLeft || hitRight)
					{
						if (hitLeft && hitRight)
						{
							nodeIdx = (t0.x < t1.x) ? node.m_leftChildIdx : node.m_leftChildIdx + 1;
							if (top < 64)
							{
								stack[top++] = (t0.x < t1.x) ? node.m_leftChildIdx + 1 : node.m_leftChildIdx;
							}
						}
						else
						{
							nodeIdx = (hitLeft) ? node.m_leftChildIdx : node.m_leftChildIdx + 1;
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
	m_nInternalNodes = nInternalNodes;
	d_bvhNodes.resize(nInternalNodes + nLeafNodes);
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

		d_flags.resize(nInternalNodes); d_flags.reset();
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
		const auto debugFittedNodes = d_bvhNodes.getData();
		printf("Test\n");
#endif
	}
}

void TwoPassLbvh::traverseBvh(Context& context)
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
	cam.m_quat = qtRotation(float4{ 0.0f, 1.0f, 0.0f, -1.57f });
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
			"../src/CommonBlocksKernel.h",
			"GenerateRays",
			std::nullopt);

		generateRaysKernel.setArgs({ d_cam.ptr(), d_rayBuffer.ptr(), width, height });
		m_timer.measure(TimerCodes::BvhBuildTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

#if _CPU																																																						
	//CPU traversal 
	const auto debugRayBuff = d_rayBuffer.getData();
	const auto debugBvhNodes = d_bvhNodes.getData();
	const auto debugTriangle = d_triangleBuff.getData();

	const u32 launchSize = width * height;
	std::vector<HitInfo> h_hitInfo;
	u8* colorBuffer = (u8*)malloc(launchSize * 4);
	memset(colorBuffer, 0, launchSize * 4);

	TraversalCPU(debugRayBuff, debugBvhNodes, debugTriangle, t, colorBuffer, width, height, m_nInternalNodes);

	stbi_write_png("test.png", width, height, 4, colorBuffer, width * 4);
	free(colorBuffer);
#else
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
			"../src/TwoPassLbvhKernel.h",
			"BvhTraversalifif",
			std::nullopt);

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}
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
			"../src/TwoPassLbvhKernel.h",
			"BvhTraversalWhile",
			std::nullopt);

		traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), m_rootNodeIdx, width, height, m_nInternalNodes });
		m_timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
	}

#endif

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
