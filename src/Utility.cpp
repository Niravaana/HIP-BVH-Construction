#include "Utility.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dependencies/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <dependencies/stbi/stb_image.h>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace BvhConstruction;

/* Call this function and check if the returned value is same as value calculated by scene extent kernel and bvh root nodes*/
bool Utility::checkLbvhRootAabb(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes)
{
	Aabb rootAabb;
	std::vector<u32> primIdxsX;
	rootAabb.reset();
	for (size_t i = 0; i < nLeafNodes; i++)
	{
		rootAabb.grow(bvhNodes[nInternalNodes + i].m_aabb);
	}

	bool isCorrect = (rootAabb.m_min == bvhNodes[rootIdx].m_aabb.m_min) && (rootAabb.m_max == bvhNodes[rootIdx].m_aabb.m_max);
	return isCorrect;
}

/* This function will do DFS and visiting all the leaf nodes collect primIdxin a vector. 
The size of this vector should match nLeaf Nodes and this vector should have unique values*/
bool Utility::checkLBvhCorrectness(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes)
{
	std::vector<u32> primIdxs;
	{
		u32 stack[32];
		int top = 0;
		stack[top++] = INVALID_NODE_IDX;
		u32 nodeIdx = rootIdx;

		while (nodeIdx != INVALID_NODE_IDX)
		{
			LbvhNode node = bvhNodes[nodeIdx];
			if (nodeIdx >= nInternalNodes)
			{
				primIdxs.push_back(node.m_leftChildIdx);
			}
			else
			{
				stack[top++] = node.m_leftChildIdx;
				stack[top++] = node.m_rightChildIdx;
			}
			nodeIdx = stack[--top];
		}
	}

	std::sort(primIdxs.begin(), primIdxs.end());
	int uniqueCount = std::unique(primIdxs.begin(), primIdxs.end()) - primIdxs.begin();

	return primIdxs.size() == nLeafNodes && uniqueCount == nLeafNodes;
}

bool Utility::checkPlocBvh2Correctness(const LbvhNode* bvhNodes, const PrimRef* leafNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes)
{
	std::vector<uint32_t> primIdxs;
	{
		uint32_t stack[32];
		int top = 0;
		stack[top++] = INVALID_NODE_IDX;
		uint32_t nodeIdx = rootIdx;

		while (nodeIdx != INVALID_NODE_IDX)
		{
			if (nodeIdx >= nInternalNodes)
			{
				primIdxs.push_back(leafNodes[nodeIdx - nInternalNodes].m_primIdx);
			}
			else
			{
				LbvhNode node = bvhNodes[nodeIdx];
				stack[top++] = node.m_leftChildIdx;
				stack[top++] = node.m_rightChildIdx;
			}
			nodeIdx = stack[--top];
		}
	}

	std::sort(primIdxs.begin(), primIdxs.end());
	int uniqueCount = std::unique(primIdxs.begin(), primIdxs.end()) - primIdxs.begin();

	return primIdxs.size() == nLeafNodes && uniqueCount == nLeafNodes;
}

bool BvhConstruction::Utility::checkLBvh4Correctness(const Bvh4Node* bvhNodes, const PrimNode* wideLeafNodes, u32 rootIdx, u32 nInternalNodes)
{
	std::vector<u32> primIdxs;
	{
		u32 stack[64];
		int top = 0;
		stack[top++] = INVALID_NODE_IDX;
		u32 nodeIdx = rootIdx;

		while (nodeIdx != INVALID_NODE_IDX)
		{
			if (nodeIdx >= nInternalNodes)
			{
				const auto leaf = wideLeafNodes[nodeIdx - nInternalNodes];
				primIdxs.push_back(leaf.m_primIdx);
			}
			else
			{
				Bvh4Node node = bvhNodes[nodeIdx];
				if(node.m_child[0] != INVALID_NODE_IDX)
					stack[top++] = node.m_child[0];
				if (node.m_child[1] != INVALID_NODE_IDX)
					stack[top++] = node.m_child[1];
				if (node.m_child[2] != INVALID_NODE_IDX)
					stack[top++] = node.m_child[2];
				if (node.m_child[3] != INVALID_NODE_IDX)
					stack[top++] = node.m_child[3];
			}
			nodeIdx = stack[--top];
		}
	}

	std::sort(primIdxs.begin(), primIdxs.end());
	int uniqueCount = std::unique(primIdxs.begin(), primIdxs.end()) - primIdxs.begin();

	return primIdxs.size() == nInternalNodes + 1 && uniqueCount == nInternalNodes + 1;

}

bool Utility::checkSahCorrectness(const SahBvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes)
{
	u32 stack[32];
	int top = 0;
	stack[top++] = INVALID_NODE_IDX;
	u32 nodeIdx = 0;
	std::vector<u32> primIdxs;

	while (nodeIdx != INVALID_NODE_IDX)
	{
		SahBvhNode node = bvhNodes[nodeIdx];
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

	std::sort(primIdxs.begin(), primIdxs.end());
	int uniqueCount = std::unique(primIdxs.begin(), primIdxs.end()) - primIdxs.begin();

	return primIdxs.size() == nLeafNodes && uniqueCount == nLeafNodes;
}

void Utility::TraversalLbvhCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height, u32 nInternalNodes)
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

				if (nodeIdx >= nInternalNodes)
				{
					Triangle& triangle = primitives[node.m_leftChildIdx];
					float3 tV0 = transform(triangle.v1, t.m_scale, t.m_quat, t.m_translation);
					float3 tV1 = transform(triangle.v2, t.m_scale, t.m_quat, t.m_translation);
					float3 tV2 = transform(triangle.v3, t.m_scale, t.m_quat, t.m_translation);

					float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);
					if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
					{
						hit.m_primIdx = node.m_leftChildIdx;
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

void Utility::TraversalSahBvhCPU(const std::vector<Ray>& rayBuff, std::vector<SahBvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height)
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

float Utility::calculateLbvhCost(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes)
{
	u32 nodeIdx = rootIdx;
	float cost = 0.0f;
	constexpr float ci = 1.0f;
	constexpr float ct = 1.0f;
	constexpr u32 nPrimsPerLeaf = 1;
	const float rootInvArea = 1.0f / bvhNodes[rootIdx].m_aabb.area();

	cost += ct; //cost of root node
	for (int i = 0; i < nInternalNodes; i++)
	{
		u32 leftChild = bvhNodes[i].m_leftChildIdx;
		if (leftChild != INVALID_NODE_IDX)
		{
			cost += ct * bvhNodes[leftChild].m_aabb.area() * rootInvArea;
		}
		u32 rightChild = bvhNodes[i].m_rightChildIdx;
		if (rightChild != INVALID_NODE_IDX)
		{
			cost += ct * bvhNodes[rightChild].m_aabb.area() * rootInvArea;
		}
	}
	for (int i = nInternalNodes; i < nLeafNodes + nInternalNodes; i++)
	{
		if (bvhNodes[i].m_leftChildIdx != INVALID_NODE_IDX)
		{
			cost += ci * bvhNodes[i].m_aabb.area() * rootInvArea;
		}
	}
	
	return cost;
}

float BvhConstruction::Utility::calculatebvh4Cost(const Bvh4Node* bvhNodes, const PrimNode* bvh4LeafNodes, Aabb* primAabbs,u32 rootIdx, u32 totalNodes, u32 nInternalNodes)
{
	u32 nodeIdx = rootIdx;
	float cost = 0.0f;
	constexpr float ci = 1.0f;
	constexpr float ct = 1.0f;
	constexpr u32 nPrimsPerLeaf = 1;
	Aabb rootAabb; 
	if(bvhNodes[rootIdx].m_child[0] != INVALID_NODE_IDX)
		rootAabb.grow(bvhNodes[rootIdx].m_aabb[0]);
	if (bvhNodes[rootIdx].m_child[1] != INVALID_NODE_IDX)
		rootAabb.grow(bvhNodes[rootIdx].m_aabb[1]);
	if (bvhNodes[rootIdx].m_child[2] != INVALID_NODE_IDX)
		rootAabb.grow(bvhNodes[rootIdx].m_aabb[2]);
	if (bvhNodes[rootIdx].m_child[3] != INVALID_NODE_IDX)
		rootAabb.grow(bvhNodes[rootIdx].m_aabb[3]);

	const float rootInvArea = 1.0f / rootAabb.area();

	cost += ct; //cost of root node
	for (int i = 0; i < totalNodes; i++)
	{
		if (bvhNodes[i].m_child[0] != INVALID_NODE_IDX && bvhNodes[i].m_child[0] < nInternalNodes)
		{
			cost += ct * bvhNodes[i].m_aabb[0].area() * rootInvArea;
		}
		if (bvhNodes[i].m_child[1] != INVALID_NODE_IDX && bvhNodes[i].m_child[1] < nInternalNodes)
		{
			cost += ct * bvhNodes[i].m_aabb[1].area() * rootInvArea;
		}
		if (bvhNodes[i].m_child[2] != INVALID_NODE_IDX && bvhNodes[i].m_child[2] < nInternalNodes)
		{
			cost += ct * bvhNodes[i].m_aabb[2].area() * rootInvArea;
		}
		if (bvhNodes[i].m_child[3] != INVALID_NODE_IDX && bvhNodes[i].m_child[3] < nInternalNodes)
		{
			cost += ct * bvhNodes[i].m_aabb[3].area() * rootInvArea;
		}
	}

	for (int i = 0; i < nInternalNodes + 1; i++)
	{
		cost += primAabbs[bvh4LeafNodes[i].m_primIdx].area() * rootInvArea;
	}
	return cost;
}

float Utility::calculateBinnedSahBvhCost(const SahBvhNode* bvhNodes, u32 rootIdx, u32 totalNodes)
{
	u32 nodeIdx = rootIdx;
	float cost = 0.0f;
	constexpr float ci = 1.0f;
	constexpr float ct = 1.0f;
	constexpr u32 nPrimsPerLeaf = 1;
	const float rootInvArea = 1.0f / bvhNodes[rootIdx].m_aabb.area();

	cost += ct; //cost of root node
	for (int i = 0; i < totalNodes; i++)
	{
		if (bvhNodes[i].m_firstChildIdx != INVALID_NODE_IDX)
		{
			u32 leftChild = bvhNodes[i].m_firstChildIdx;
			cost += ((SahBvhNode::isLeafNode(bvhNodes[leftChild])) ? ci : ct) * bvhNodes[leftChild].m_aabb.area() * rootInvArea;
		}
		if (bvhNodes[i].m_firstChildIdx + 1 != INVALID_NODE_IDX)
		{
			u32 rightChild = bvhNodes[i].m_firstChildIdx + 1;
			cost += ((SahBvhNode::isLeafNode(bvhNodes[rightChild])) ? ci : ct) * bvhNodes[rightChild].m_aabb.area() * rootInvArea;
		}
	}
	return cost;
}

void Utility::generateTraversalHeatMap(std::vector<u32> rayCounter, u32 width, u32 height)
{
	u32 max = 0;
	for (int i = 0; i < rayCounter.size(); i++)
	{
		if (rayCounter[i] > max)
			max = rayCounter[i];
	}
	const u32 launchSize = width * height;
	std::vector<HitInfo> h_hitInfo;
	u8* colorBuffer = (u8*)malloc(launchSize * 4);
	memset(colorBuffer, 0, launchSize * 4);
	std::vector<float3> debugColors;

	for (int gIdx = 0; gIdx < width; gIdx++)
	{
		for (int gIdy = 0; gIdy < height; gIdy++)
		{
			u32 index = gIdx * width + gIdy;
			colorBuffer[index * 4 + 0] = (rayCounter[index] / (float)max) * 150;
			colorBuffer[index * 4 + 1] = (rayCounter[index] / (float)max) * 255;
			colorBuffer[index * 4 + 2] = 255;
			colorBuffer[index * 4 + 3] = 255;
			float3 col = { colorBuffer[index * 4 + 0], colorBuffer[index * 4 + 1], 0.0f };
			debugColors.push_back(col);
		}
	}

	stbi_write_png("colorMap.png", width, height, 4, colorBuffer, width * 4);
	free(colorBuffer);
}

void Utility::doEarlySplitClipping(std::vector<Triangle>& inputPrims, std::vector<PrimRef>& outPrimRefs, float saMax)
{
	std::queue<PrimRef> taskQueue;
	for (int i = 0; i < inputPrims.size(); i++)
	{
		Aabb primAabb;
		primAabb.grow(inputPrims[i].v1);
		primAabb.grow(inputPrims[i].v2);
		primAabb.grow(inputPrims[i].v3);
		PrimRef ref = { i, primAabb };

		taskQueue.push(ref);
	}

	while (!taskQueue.empty())
	{
		PrimRef ref = taskQueue.front(); taskQueue.pop();

		if (ref.m_aabb.area() <= saMax)
		{
			outPrimRefs.push_back(ref);
		}
		else
		{
			const auto extent = ref.m_aabb.extent();
			const int dim = ref.m_aabb.maximumExtentDim();
			const auto centre = ref.m_aabb.center();
			const auto offset = ref.m_aabb.m_min + (extent - centre);

			float3 lMin = ref.m_aabb.m_min;
			float3 lMax;
			if (dim == 0)
			{
				lMax.x = centre.x;
				lMax.y = ref.m_aabb.m_max.y;
				lMax.z = ref.m_aabb.m_max.z;
			}
			if (dim == 1)
			{
				lMax.x = ref.m_aabb.m_max.x;
				lMax.y = centre.y;
				lMax.z = ref.m_aabb.m_max.z;
			}
			if (dim == 2)
			{
				lMax.x = ref.m_aabb.m_max.x;
				lMax.y = ref.m_aabb.m_max.y;
				lMax.z = centre.z;
			}

			Aabb L = { lMin, lMax };

			float3 RMin;
			float3 RMax = ref.m_aabb.m_max;
			if (dim == 0)
			{
				RMin.x = centre.x;
				RMin.y = ref.m_aabb.m_min.y;
				RMin.z = ref.m_aabb.m_min.z;
			}
			if (dim == 1)
			{
				RMin.x = ref.m_aabb.m_min.x;
				RMin.y = centre.y;
				RMin.z = ref.m_aabb.m_min.z;
			}
			if (dim == 2)
			{
				RMin.x = ref.m_aabb.m_min.x;
				RMin.y = ref.m_aabb.m_min.y;
				RMin.z = centre.z;
			}

			Aabb R = { RMin, RMax };

			PrimRef LRef = { ref.m_primIdx, L };
			PrimRef RRef = { ref.m_primIdx, R };

			taskQueue.push(LRef);
			taskQueue.push(RRef);
		}
	}
}

void Utility::collapseBvh2toBvh4(const std::vector<LbvhNode>& bvh2Nodes, std::vector<Bvh4Node>& bvh4Nodes, std::vector<PrimNode> bvh4LeafNodes, std::vector<uint2>& taskQ, u32 taskCount, u32& bvh8InternalNodeOffset, u32 nBvh2InternalNodes, u32 nBvh2LeafNodes)
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


void BvhConstruction::MeshLoader::loadScene(const std::string & filename, const std::string & mtlBaseDir, std::vector<Triangle>& trianglesOut)
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