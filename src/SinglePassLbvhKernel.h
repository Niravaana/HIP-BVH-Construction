
#include <src/Common.h>

using namespace BvhConstruction;

extern "C" __global__ void InitBvhNodes(
	const Triangle* __restrict__ primitives,
	LbvhNode* __restrict__ bvhNodes,
	const u32* __restrict__ primIdx,
	const u32 nInternalNodes,
	const u32 nLeafNodes)
{
	unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (gIdx < nLeafNodes)
	{
		const u32 nodeIdx = gIdx + nInternalNodes;
		u32 idx = primIdx[gIdx];
		LbvhNode& node = bvhNodes[nodeIdx];
		node.m_aabb.reset();
		node.m_aabb.grow(primitives[idx].v1); node.m_aabb.grow(primitives[idx].v2); node.m_aabb.grow(primitives[idx].v3);
		node.m_leftChildIdx = idx;
		node.m_rightChildIdx = INVALID_NODE_IDX;
	}

	if (gIdx < nInternalNodes)
	{
		LbvhNode& node = bvhNodes[gIdx];
		node.m_aabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
	}
}

DEVICE uint64_t findHighestDiffBit(const u32* __restrict__ mortonCodes, int i, int j, int n)
{
	if (j < 0 || j >= n) return ~0ull;
	const uint64_t a = (static_cast<uint64_t>(mortonCodes[i]) << 32ull) | i;
	const uint64_t b = (static_cast<uint64_t>(mortonCodes[j]) << 32ull) | j;
	return a ^ b;
}

DEVICE int findParent(
	LbvhNode* __restrict__ bvhNodes,
	uint2* spans,
	const u32* __restrict__ mortonCodes,
	u32 currentNodeIdx,
	int i,
	int j,
	int n)
{
	if (i == 0 && j == n) return INVALID_NODE_IDX;
	if (i == 0 || (j != n && findHighestDiffBit(mortonCodes, j - 1, j, n) < findHighestDiffBit(mortonCodes, i - 1, i, n)))
	{
		bvhNodes[j - 1].m_leftChildIdx = currentNodeIdx;
		spans[j - 1].x = i;
		return j - 1;
	}
	else
	{
		bvhNodes[i - 1].m_rightChildIdx = currentNodeIdx;
		spans[i - 1].y = j;
		return i - 1;
	}
}

extern "C" __global__ void BvhBuildAndFit(
	LbvhNode* __restrict__ bvhNodes,
	int* __restrict__ bvhNodeCounter,
	uint2* __restrict__ spans,
	const u32* __restrict__ mortonCodes,
	u32 nLeafNodes,
	u32 nInternalNodes)
{
	int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= nLeafNodes) return;

	int i = gIdx;
	int j = gIdx + 1;

	int parentIdx = findParent(bvhNodes, spans, mortonCodes, nInternalNodes + gIdx, i, j, nLeafNodes);
	gIdx = parentIdx;

	while (atomicAdd(&bvhNodeCounter[gIdx], 1) > 0)
	{
		__threadfence();

		LbvhNode& node = bvhNodes[gIdx];
		uint2 span = spans[gIdx];

		node.m_aabb = merge(bvhNodes[node.m_leftChildIdx].m_aabb, bvhNodes[node.m_rightChildIdx].m_aabb);

		parentIdx = findParent(bvhNodes, spans, mortonCodes, gIdx, span.x, span.y, nLeafNodes);

		if (parentIdx == INVALID_NODE_IDX)
		{
			bvhNodeCounter[nLeafNodes - 1] = gIdx; //Saving root node;
			break;
		}

		gIdx = parentIdx;

		__threadfence();
	}
}
