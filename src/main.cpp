#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <dependencies/Orochi/Orochi/GpuMemory.h>

#include <src/Error.h>
#include <src/Kernel.h>
#include <src/Timer.h>
#include <src/Context.h>
#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "SinglePassLbvh.h"
#include "TwoPassLbvh.h"
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

int main(int argc, char* argv[])
{
	try
	{
		Context context;
		Timer timer;
		TwoPassLbvh bvh;

		std::vector<Triangle> triangles;
		//loadScene("../src/meshes/cornellbox/cornellbox.obj", "../src/meshes/cornellbox/", triangles);
		loadScene("../src/meshes/sponza/sponza.obj", "../src/bunny/sponza/", triangles);

		bvh.build(context, triangles);
		bvh.traverseBvh(context);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}