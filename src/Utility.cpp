#include "Utility.h"
#include <algorithm>
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
				primIdxs.push_back(node.m_primIdx);
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

void Utility::TraversalLbvhCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height)
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
	for (int i = 0; i < nLeafNodes + nInternalNodes; i++)
	{
		if (bvhNodes[i].m_leftChildIdx != INVALID_NODE_IDX)
		{
			u32 leftChild = bvhNodes[i].m_leftChildIdx;
			cost += ((leftChild >= nInternalNodes) ? ci : ct)* bvhNodes[leftChild].m_aabb.area()* rootInvArea;
		}
		if (bvhNodes[i].m_rightChildIdx != INVALID_NODE_IDX)
		{
			u32 rightChild = bvhNodes[i].m_rightChildIdx;
			cost += ((rightChild >= nInternalNodes) ? ci : ct) * bvhNodes[rightChild].m_aabb.area() * rootInvArea;
		}
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
