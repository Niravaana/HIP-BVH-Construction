#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <dependencies/Orochi/Orochi/GpuMemory.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dependencies/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <dependencies/stbi/stb_image.h>
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
	BvhBuildTime,
	TraversalTime
};



HitInfo TraversalCPU(const Ray& ray, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t)
{
	u32 nodeIdx = 0;
	u32 top = 0;
	u32 stack[64];
	stack[top++] = INVALID_NODE_IDX;
	HitInfo hit;
	
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
			if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f &&  itr.w < hit.m_t)
			{
				hit.m_primIdx = node.m_primIdx;
				hit.m_t = itr.w;
				hit.m_uv = {itr.x, itr.y};
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
						stack[top++] =  (t0.x < t1.x) ? node.m_rightChildIdx : node.m_leftChildIdx;
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
	return hit;
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
			OrochiUtils::waitForCompletion();
		}

#if _DEBUG
		const auto debugFittedNodes = d_bvhNodes.getData();
#endif

		//set transformation for the scene (fixed currently for cornell box)
		Transformation t;
		t.m_translation = float3{ 0.0f, 0.0f, -3.0f };
		t.m_scale = float3{ 1.0f, 1.0f, 1.0f };
		t.m_quat = {0.0f, 1.0f, 0.0f, 1.0f}; qtGetIdentity();
		Oro::GpuMemory<Transformation> d_transformations(1); d_transformations.reset();
		OrochiUtils::copyHtoD(d_transformations.ptr(), &t, 1);

		//create camera 
		Camera cam;
		cam.m_eye = float4{ 0.0f, 2.5f, 5.8f, 0.0f };
		cam.m_quat = qtRotation(float4{ 0.0f, 0.0f, 1.0f, -1.57f });
		cam.m_fov = 45.0f *  Pi / 180.f;
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
				orochiDevice,
				"../src/LbvhKernel.h",
				"GenerateRays",
				std::nullopt);

			generateRaysKernel.setArgs({ d_cam.ptr(), d_rayBuffer.ptr(), width, height});
			timer.measure(TimerCodes::BvhBuildTime, [&]() { generateRaysKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
		}

#if _DEBUG
		const auto debugRayBuff = d_rayBuffer.getData();
		const auto debugBvhNodes = d_bvhNodes.getData();
#endif


#if _CPU
		//CPU traversal 
		const u32 launchSize = width * height;
		std::vector<HitInfo> h_hitInfo;
		u8* dst = (u8*)malloc(launchSize * 4);
		memset(dst, 0, launchSize * 4);

		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				u32 index = i + j * width;
				HitInfo hit = TraversalCPU(debugRayBuff[index], debugBvhNodes, debugTriangle, t);
				if (hit.m_primIdx != INVALID_PRIM_IDX)
				{
					dst[index * 4 + 0] = (hit.m_t/30.0f) * 255;
					dst[index * 4 + 1] = (hit.m_t / 30.0f) * 255;
					dst[index * 4 + 2] = (hit.m_t / 30.0f) * 255;
					dst[index * 4 + 3] = 255;
				}
			}
		}

		stbi_write_png("test.png", width, height, 4, dst, width * 4);
		free(dst);
#else

		Oro::GpuMemory<u8> d_colorBuffer(width* height * 4); d_colorBuffer.reset();
		Oro::GpuMemory<HitInfo> d_hitInfoBuff(width* height);
		//Traversal kernel
		{
			const u32 blockSizeX = 8;
			const u32 blockSizeY = 8;
			const u32 gridSizeX = (width + blockSizeX - 1) / blockSizeX;
			const u32 gridSizeY = (height + blockSizeY - 1) / blockSizeY;
			Kernel traversalKernel;

			buildKernelFromSrc(
				traversalKernel,
				orochiDevice,
				"../src/LbvhKernel.h",
				"BvhTraversal",
				std::nullopt);

			traversalKernel.setArgs({ d_rayBuffer.ptr(), d_triangleBuff.ptr(), d_bvhNodes.ptr(), d_transformations.ptr(), d_colorBuffer.ptr(), d_hitInfoBuff.ptr(), width, height });
			timer.measure(TimerCodes::TraversalTime, [&]() { traversalKernel.launch(gridSizeX, gridSizeY, 1, blockSizeX, blockSizeY, 1); });
		}

		stbi_write_png("test.png", width, height, 4, d_colorBuffer.getData().data(), width * 4);
#endif 

		const auto tes = d_hitInfoBuff.getData();
		const auto xx = d_transformations.getData();

		CHECK_ORO(oroCtxDestroy(orochiCtxt));
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}