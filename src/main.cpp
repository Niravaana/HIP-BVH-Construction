#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <dependencies/Orochi/Orochi/GpuMemory.h>
#include <ParallelPrimitives/RadixSort.h>
#include <src/Error.h>
#include <src/Kernel.h>
#include <src/Timer.h>
#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace BvhConstruction;

void loadScene(const std::string& filename,	const std::string& mtlBaseDir, std::vector<Triangle>& trianglesOut )
{
	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &err, filename.c_str(), mtlBaseDir.c_str());

	if (!warning.empty())
	{
		std::cerr << "OBJ Loader WARN : " << warning << std::endl;
	}

	if (!err.empty())
	{
		std::cerr << "OBJ Loader ERROR : " << err << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!ret)
	{
		std::cerr << "Failed to load obj file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (shapes.empty())
	{
		std::cerr << "No shapes in obj file (run 'git lfs fetch' and 'git lfs pull' in 'test/common/meshes/lfs')" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::vector<int>		materialIndices; // material ids for all instances
	std::vector<u32>	instanceMask;
	std::vector<float3>		allVertices;
	std::vector<float3>		allNormals;
	std::vector<u32>	allIndices;

	// Prefix sum to calculate the offsets in to global vert,index and material buffer
	int						 vertexPrefixSum = 0;
	int						 normalPrefixSum = 0;
	int						 indexPrefixSum = 0;
	std::vector<int>		 indicesOffsets;
	std::vector<int>		 verticesOffsets;
	std::vector<int>		 normalsOffsets;

	indicesOffsets.resize(shapes.size());
	verticesOffsets.resize(shapes.size());
	normalsOffsets.resize(shapes.size());

	auto convert = [](const tinyobj::real_t c[3]) -> float4 { return float4{ c[0], c[1], c[2], 0.0f }; };
	
	auto compare = [](const tinyobj::index_t& a, const tinyobj::index_t& b) {
		if (a.vertex_index < b.vertex_index) return true;
		if (a.vertex_index > b.vertex_index) return false;

		if (a.normal_index < b.normal_index) return true;
		if (a.normal_index > b.normal_index) return false;

		if (a.texcoord_index < b.texcoord_index) return true;
		if (a.texcoord_index > b.texcoord_index) return false;

		return false;
	};

	for (size_t i = 0; i < shapes.size(); ++i)
	{
		std::vector<float3>									 vertices;
		std::vector<float3>									 normals;
		std::vector<u32>								 indices;
		float3* v = reinterpret_cast<float3*>(attrib.vertices.data());
		std::map<tinyobj::index_t, int, decltype(compare)> knownIndex(compare);

		for (size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++)
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

			if (knownIndex.find(idx0) != knownIndex.end())
			{
				indices.push_back(knownIndex[idx0]);
			}
			else
			{
				knownIndex[idx0] = static_cast<int>(vertices.size());
				indices.push_back(knownIndex[idx0]);
				vertices.push_back(v[idx0.vertex_index]);
				normals.push_back(v[idx0.normal_index]);
			}

			if (knownIndex.find(idx1) != knownIndex.end())
			{
				indices.push_back(knownIndex[idx1]);
			}
			else
			{
				knownIndex[idx1] = static_cast<int>(vertices.size());
				indices.push_back(knownIndex[idx1]);
				vertices.push_back(v[idx1.vertex_index]);
				normals.push_back(v[idx1.normal_index]);
			}

			if (knownIndex.find(idx2) != knownIndex.end())
			{
				indices.push_back(knownIndex[idx2]);
			}
			else
			{
				knownIndex[idx2] = static_cast<int>(vertices.size());
				indices.push_back(knownIndex[idx2]);
				vertices.push_back(v[idx2.vertex_index]);
				normals.push_back(v[idx2.normal_index]);
			}

			materialIndices.push_back(shapes[i].mesh.material_ids[face]);
		}

		verticesOffsets[i] = vertexPrefixSum;
		vertexPrefixSum += static_cast<int>(vertices.size());
		indicesOffsets[i] = indexPrefixSum;
		indexPrefixSum += static_cast<int>(indices.size());
		normalsOffsets[i] = normalPrefixSum;
		normalPrefixSum += static_cast<int>(normals.size());

		allVertices.insert(allVertices.end(), vertices.begin(), vertices.end());
		allNormals.insert(allNormals.end(), normals.begin(), normals.end());
		allIndices.insert(allIndices.end(), indices.begin(), indices.end());
	}

	for (size_t i = 0; i < shapes.size(); ++i)
	{
		uint32_t* indices = &allIndices[indicesOffsets[i]];
		float3* vertices = &allVertices[verticesOffsets[i]];
		u32 indexCount = 3 * static_cast<u32>(shapes[i].mesh.num_face_vertices.size());

		for (int j = 0; j < indexCount; j += 3)
		{
			const u32 idx1 = indices[j];
			const u32 idx2 = indices[j + 1];
			const u32 idx3 = indices[j + 2];

			trianglesOut.push_back(Triangle{ vertices[idx1], vertices[idx2], vertices[idx3] });
		}
	}
}

enum TimerCodes
{
	CalculateCentroidExtentsTime,
	CalculateMortonCodesTime,
	SortingTime,
	BvhBuildTime
};

HOST_DEVICE INLINE uint32_t lcg(uint32_t& seed)
{
	const uint32_t LCG_A = 1103515245u;
	const uint32_t LCG_C = 12345u;
	const uint32_t LCG_M = 0x00FFFFFFu;
	seed = (LCG_A * seed + LCG_C);
	return seed & LCG_M;
}

HOST_DEVICE INLINE float randf(uint32_t& seed)
{
	return (static_cast<float>(lcg(seed)) / static_cast<float>(0x01000000));
}

template <uint32_t N>
HOST_DEVICE INLINE uint2 tea(uint32_t val0, uint32_t val1)
{
	uint32_t v0 = val0;
	uint32_t v1 = val1;
	uint32_t s0 = 0;

	for (uint32_t n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return { v0, v1 };
}

void GenerateRays(const Camera* cam, Ray* raysBuffOut, const u32 width, const u32 height)
{

	for (int gIdx = 0; gIdx < width; gIdx++)
	{

		for (int gIdy = 0; gIdy < height; gIdy++)
		{
			//if (gIdx >= width) return;
			//if (gIdy >= height) return;

			bool isMultiSamples = false;
			float  fov = cam->m_fov;
			float2 sensorSize;
			unsigned int seed = tea<16>(gIdx + gIdy * width, 0).x;

			sensorSize.x = 0.024f * (width / static_cast<float>(height));
			sensorSize.y = 0.024f;
			float		 offset = (isMultiSamples) ? randf(seed) : 0.5f;
			const float2 xy = float2{ ((float)gIdx + offset) / width, ((float)gIdy + offset) / height } - float2{ 0.5f, 0.5f };
			float3		 dir = float3{ xy.x * sensorSize.x, xy.y * sensorSize.y, sensorSize.y / (2.f * tan(fov / 2.f)) };

			const float3 holDir = qtRotate(cam->m_quat, float3{ 1.0f, 0.0f, 0.0f });
			const float3 upDir = qtRotate(cam->m_quat, float3{ 0.0f, -1.0f, 0.0f });
			const float3 viewDir = qtRotate(cam->m_quat, float3{ 0.0f, 0.0f, -1.0f });
			dir = normalize(dir.x * holDir + dir.y * upDir + dir.z * viewDir);

			{
				Ray& r = raysBuffOut[gIdx * height + gIdy];
				r.m_origin = float3{ cam->m_eye.x, cam->m_eye.y, cam->m_eye.z };
				float4 direction = cam->m_eye + float4{ dir.x * cam->m_far, dir.y * cam->m_far, dir.z * cam->m_far, 0.0f };
				r.m_direction = normalize(float3{ direction.x, direction.y, direction.z });
			}
		}
	}
}

int main(int argc, char* argv[])
{
	try
	{
		oroDevice	orochiDevice;
		oroCtx		orochiCtxt;
		Timer timer;
		
		std::vector<Triangle> triangles;
		loadScene("../src/meshes/cornellbox/cornellbox.obj", "../src/meshes/cornellbox/", triangles);
		


		CHECK_ORO((oroError)oroInitialize((oroApi)(ORO_API_HIP), 0));
		CHECK_ORO(oroInit(0));
		CHECK_ORO(oroDeviceGet(&orochiDevice, 0)); // deviceId should be taken from user?
		CHECK_ORO(oroCtxCreate(&orochiCtxt, 0, orochiDevice));
		oroDeviceProp props;
		CHECK_ORO(oroGetDeviceProperties(&props, orochiDevice));
		std::cout << "Executing on '" << props.name << "'" << std::endl;

		const size_t primitiveCount = triangles.size();
		Oro::GpuMemory<Triangle> d_triangleBuff(primitiveCount); d_triangleBuff.reset();
		Oro::GpuMemory<Aabb> d_triangleAabb(primitiveCount); d_triangleAabb.reset(); //ToDo we might not need it.
		OrochiUtils::copyHtoD(d_triangleBuff.ptr(), triangles.data(), triangles.size());

		Oro::GpuMemory<Aabb> d_sceneExtents(1); d_sceneExtents.reset();
		{
			Kernel centroidExtentsKernel;

			buildKernelFromSrc(
				centroidExtentsKernel,
				orochiDevice,
				"../src/LbvhKernel.h",
				"CalculateSceneExtents",
				std::nullopt);

			centroidExtentsKernel.setArgs({d_triangleBuff.ptr(), d_triangleAabb.ptr(), d_sceneExtents.ptr(), primitiveCount });
			timer.measure(TimerCodes::CalculateCentroidExtentsTime, [&]() { centroidExtentsKernel.launch(primitiveCount); });
		}

#if _DEBUG
		const auto debugTriangle = d_triangleBuff.getData();
		const auto debugAabb = d_triangleAabb.getData();
		const auto debugExtent = d_sceneExtents.getData()[0];
#endif

		Oro::GpuMemory<u32> d_mortonCodeKeys(primitiveCount); d_mortonCodeKeys.reset();
		Oro::GpuMemory<u32> d_mortonCodeValues(primitiveCount); d_mortonCodeValues.reset();
		Oro::GpuMemory<u32> d_sortedMortonCodeKeys(primitiveCount); d_sortedMortonCodeKeys.reset();
		Oro::GpuMemory<u32> d_sortedMortonCodeValues(primitiveCount); d_sortedMortonCodeValues.reset();
		{
			Kernel calulateMortonCodesKernel;

			buildKernelFromSrc(
				calulateMortonCodesKernel,
				orochiDevice,
				"../src/LbvhKernel.h",
				"CalculateMortonCodes",
				std::nullopt);

			calulateMortonCodesKernel.setArgs({ d_triangleAabb.ptr(), d_sceneExtents.ptr() , d_mortonCodeKeys.ptr(), d_mortonCodeValues.ptr(), primitiveCount});
			timer.measure(TimerCodes::CalculateMortonCodesTime, [&]() { calulateMortonCodesKernel.launch(primitiveCount); });
		}

#if _DEBUG
		const auto debugMortonCodes = d_mortonCodeKeys.getData();
#endif
		{
			OrochiUtils oroUtils;
			Oro::RadixSort sort(orochiDevice, oroUtils, 0, "../dependencies/Orochi/ParallelPrimitives/RadixSortKernels.h", "../dependencies/Orochi");

			Oro::RadixSort::KeyValueSoA srcGpu{};
			Oro::RadixSort::KeyValueSoA dstGpu{};
			static constexpr auto startBit{ 0 };
			static constexpr auto endBit{ 32 };
			static constexpr auto stream = 0;

			srcGpu.key = d_mortonCodeKeys.ptr();
			srcGpu.value = d_mortonCodeValues.ptr();

			dstGpu.key = d_sortedMortonCodeKeys.ptr();
			dstGpu.value = d_sortedMortonCodeValues.ptr();

			timer.measure(SortingTime, [&]() {
				sort.sort(srcGpu, dstGpu, static_cast<int>(primitiveCount), startBit, endBit, stream); });
		}

#if _DEBUG
		const auto debugSortedMortonCodes = d_sortedMortonCodeKeys.getData();
		const auto debugSortedMortonCodesVal = d_sortedMortonCodeValues.getData();
#endif

		const u32 nLeafNodes = primitiveCount;
		const u32 nInternalNodes = nLeafNodes - 1;
		Oro::GpuMemory<LbvhNode> d_bvhNodes(nInternalNodes + nLeafNodes);
		{
			{
				Kernel initBvhNodesKernel;

				buildKernelFromSrc(
					initBvhNodesKernel,
					orochiDevice,
					"../src/LbvhKernel.h",
					"InitBvhNodes",
					std::nullopt);

				initBvhNodesKernel.setArgs({ d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_sortedMortonCodeValues.ptr(), nInternalNodes, nLeafNodes});
				timer.measure(TimerCodes::BvhBuildTime, [&]() { initBvhNodesKernel.launch(nLeafNodes); });
			}

#if _DEBUG
			const auto debugNodes = d_bvhNodes.getData();
#endif

			{
				Kernel bvhBuildKernel;

				buildKernelFromSrc(
					bvhBuildKernel,
					orochiDevice,
					"../src/LbvhKernel.h",
					"BvhBuild",
					std::nullopt);

				bvhBuildKernel.setArgs({ d_bvhNodes.ptr(), d_sortedMortonCodeKeys.ptr(), nLeafNodes, nInternalNodes });
				timer.measure(TimerCodes::BvhBuildTime, [&]() { bvhBuildKernel.launch(nInternalNodes); });
			}

#if _DEBUG
			const auto debugBuiltNodes = d_bvhNodes.getData();
#endif

			Oro::GpuMemory<u32> d_flags(nInternalNodes); d_flags.reset();
			{
				Kernel fitBvhNodesKernel;

				buildKernelFromSrc(
					fitBvhNodesKernel,
					orochiDevice,
					"../src/LbvhKernel.h",
					"FitBvhNodes",
					std::nullopt);

				fitBvhNodesKernel.setArgs({ d_bvhNodes.ptr(), d_flags.ptr(), nLeafNodes, nInternalNodes });
				timer.measure(TimerCodes::BvhBuildTime, [&]() { fitBvhNodesKernel.launch(nLeafNodes); });
			}
		}

#if _DEBUG
		const auto debugFittedNodes = d_bvhNodes.getData();
#endif

		//set transformation for the scene (fixed currently for cornell box)
		Transformation t;
		t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
		t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
		Oro::GpuMemory<Transformation> d_transformations(1); d_transformations.reset();
		OrochiUtils::copyHtoD(d_transformations.ptr(), &t, 1);

		//create camera 
		Camera cam;
		cam.m_eye = float4{ 0.0f, 2.5f, 5.8f, 0.0f };
		cam.m_quat = qtRotation(float4{ 0.0f, 0.0f, 1.0f, 0.0f });
		cam.m_fov = 45.0f *  Pi / 180.f;
		cam.m_near = 0.0f;
		cam.m_far = 100000.0f;
		Oro::GpuMemory<Camera> d_cam(1); d_cam.reset();
		OrochiUtils::copyHtoD(d_cam.ptr(), &cam, 1);

		constexpr u32 width = 1280;
		constexpr u32 height = 720;
		Oro::GpuMemory<Ray> d_rayBuffer(width* height); d_rayBuffer.reset();

		/*std::vector<Ray> h_ray(width* height);
		GenerateRays(&cam, h_ray.data(), width, height);*/

		//generate rays
		{
			const u32 blockSizeX = 64;
			const u32 blockSizeY = 64;
			const u32 gridSizeX = (width + blockSizeX - 1) / blockSizeX;
			const u32 gridSizeY = (height + blockSizeY - 1) / blockSizeY;
			Kernel generateRaysKernel;

			buildKernelFromSrc(
				generateRaysKernel,
				orochiDevice,
				"../src/LbvhKernel.h",
				"GenerateRays",
				std::nullopt);

			generateRaysKernel.setArgs({ d_cam.ptr(), d_rayBuffer.ptr(), width, height});
			timer.measure(TimerCodes::BvhBuildTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
		}

#if _DEBUG
		const auto debugRayBuff = d_rayBuffer.getData();
#endif

		CHECK_ORO(oroCtxDestroy(orochiCtxt));
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}