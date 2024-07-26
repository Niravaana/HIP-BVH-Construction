#include <src/Common.h>

using namespace BvhConstruction;

DEVICE INLINE int countCommonPrefixBits(const u32 lhs, const u32 rhs)
{
	return __clz(lhs ^ rhs);
}

DEVICE INLINE int countCommonPrefixBits(const u32 lhs, const u32 rhs, const u32 i, const u32 j, const u32 n)
{
	if (j < 0 || j >= n) return ~0ull;

	const u64 a = (static_cast<u64>(lhs) << 32ull) | i;
	const u64 b = (static_cast<u64>(rhs) << 32ull) | j;

	return __clzll(a ^ b);
}

DEVICE uint2 determineRange(const u32* __restrict__ mortonCode, const u32 nLeafNodes, u32 idx)
{
	if (idx == 0)
	{
		return { 0, nLeafNodes - 1 };
	}

	// determine direction of the range
	const u32 nodeCode = mortonCode[idx];

	const int L_delta = (nodeCode == mortonCode[idx - 1]) ? countCommonPrefixBits(nodeCode, mortonCode[idx - 1], idx, idx - 1, nLeafNodes) : countCommonPrefixBits(nodeCode, mortonCode[idx - 1]);
	const int R_delta = (nodeCode == mortonCode[idx + 1]) ? countCommonPrefixBits(nodeCode, mortonCode[idx + 1], idx, idx + 1, nLeafNodes) : countCommonPrefixBits(nodeCode, mortonCode[idx + 1]);
	const int d = (R_delta > L_delta) ? 1 : -1;

	//// Compute upper bound for the length of the range
	const int deltaMin = (L_delta < R_delta) ? L_delta : R_delta;
	int lMax = 2;
	int delta = -1;
	int i_tmp = idx + d * lMax;
	if (0 <= i_tmp && i_tmp < nLeafNodes)
	{
		delta = (nodeCode == mortonCode[i_tmp]) ? countCommonPrefixBits(nodeCode, mortonCode[i_tmp], idx, i_tmp, nLeafNodes) : countCommonPrefixBits(nodeCode, mortonCode[i_tmp]);
	}
	while (delta > deltaMin)
	{
		lMax <<= 1;
		i_tmp = idx + d * lMax;
		delta = -1;
		if (0 <= i_tmp && i_tmp < nLeafNodes)
		{
			delta = (nodeCode == mortonCode[i_tmp]) ? countCommonPrefixBits(nodeCode, mortonCode[i_tmp], idx, i_tmp, nLeafNodes) : countCommonPrefixBits(nodeCode, mortonCode[i_tmp]);
		}
	}

	// Find the other end by binary search
	int l = 0;
	int t = lMax >> 1;
	while (t > 0)
	{
		i_tmp = idx + (l + t) * d;
		delta = -1;
		if (0 <= i_tmp && i_tmp < nLeafNodes)
		{
			delta = (nodeCode == mortonCode[i_tmp]) ? countCommonPrefixBits(nodeCode, mortonCode[i_tmp], idx, i_tmp, nLeafNodes) : countCommonPrefixBits(nodeCode, mortonCode[i_tmp]);
		}
		if (delta > deltaMin)
		{
			l += t;
		}
		t >>= 1;
	}

	u32 jdx = idx + l * d;
	if (d < 0)
	{
		return { jdx, idx };
	}
	return { idx, jdx };
}

DEVICE u32 findSplit(const u32* __restrict__ mortonCode, const u32 nLeafNodes, const u32 first, const u32 last)
{
	const u32 firstCode = mortonCode[first];
	const u32 lastCode = mortonCode[last];
	/*if (firstCode == lastCode)
	{
		return (first + last) >> 1;
	}*/
	const u32 deltaNode = (firstCode == lastCode) ? countCommonPrefixBits(firstCode, lastCode, first, last, nLeafNodes) :  countCommonPrefixBits(firstCode, lastCode);

	// binary search
	int split = first;
	int stride = last - first;
	do
	{
		stride = (stride + 1) >> 1;
		const int middle = split + stride;
		if (middle < last)
		{
			const u32 delta = (firstCode == mortonCode[middle]) ? countCommonPrefixBits(firstCode, mortonCode[middle], first, middle, nLeafNodes) : countCommonPrefixBits(firstCode, mortonCode[middle]);
			if (delta > deltaNode)
			{
				split = middle;
			}
		}
	} while (stride > 1);

	return split;
}

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
		node.m_primIdx = idx;
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		node.m_parentIdx = INVALID_NODE_IDX;
	}
	if (gIdx < nInternalNodes)
	{
		LbvhNode& node = bvhNodes[gIdx];
		node.m_aabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		node.m_parentIdx = INVALID_NODE_IDX;
	}
}

extern "C" __global__ void BvhBuild(
	LbvhNode* __restrict__ bvhNodes,
	const u32* __restrict__ mortonCodes,
	u32 nLeafNodes,
	u32 nInternalNodes)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= nInternalNodes) return;
	//determine range 
	uint2 range = determineRange(mortonCodes, nLeafNodes, gIdx);
	//determine split
	const u32 split = findSplit(mortonCodes, nLeafNodes, range.x, range.y);
	//create nodes and bvh
	u32 leftChildIdx = (split == range.x) ? split + nInternalNodes : split;
	u32 rightChildIdx = (split + 1 == range.y) ? (split + 1 + nInternalNodes) : split + 1;
	bvhNodes[gIdx].m_leftChildIdx = leftChildIdx;
	bvhNodes[gIdx].m_rightChildIdx = rightChildIdx;
	bvhNodes[gIdx].m_primIdx = INVALID_PRIM_IDX;
	bvhNodes[leftChildIdx].m_parentIdx = gIdx;
	bvhNodes[rightChildIdx].m_parentIdx = gIdx;
}
extern "C" __global__ void FitBvhNodes(LbvhNode* __restrict__ bvhNodes, u32* flags, u32 nLeafNodes, u32 nInternalNodes)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= nLeafNodes) return;
	int idx = nInternalNodes + gIdx;
	u32 parent = bvhNodes[idx].m_parentIdx;
	while (atomicAdd(&flags[parent], 1) > 0)
	{
		__threadfence();
		{
			u32 leftChildIdx = bvhNodes[parent].m_leftChildIdx;
			u32 rightChildIdx = bvhNodes[parent].m_rightChildIdx;
			bvhNodes[parent].m_aabb = merge(bvhNodes[leftChildIdx].m_aabb, bvhNodes[rightChildIdx].m_aabb);
			parent = bvhNodes[parent].m_parentIdx;
			if (parent == INVALID_NODE_IDX) break;
		}
		__threadfence();
	}
}
