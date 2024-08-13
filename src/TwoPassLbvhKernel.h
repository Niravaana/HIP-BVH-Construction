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
	Bvh2Node* __restrict__ bvhNodes,
	u32* __restrict__ parentIdxs,
	const u32* __restrict__ primIdx,
	const u32 nLeafNodes,
	const u32 nInternalNodes
	)
{
	unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (gIdx < nLeafNodes)
	{
		const u32 nodeIdx = gIdx + nInternalNodes;
		u32 idx = primIdx[gIdx];
		Bvh2Node& node = bvhNodes[nodeIdx];
		node.m_aabb.reset();
		node.m_aabb.grow(primitives[idx].v1); node.m_aabb.grow(primitives[idx].v2); node.m_aabb.grow(primitives[idx].v3);
		node.m_leftChildIdx = idx;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		parentIdxs[nodeIdx] = INVALID_NODE_IDX;
	}
	if (gIdx < nInternalNodes)
	{
		Bvh2Node& node = bvhNodes[gIdx];
		node.m_aabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		parentIdxs[gIdx] = INVALID_NODE_IDX;
	}
}

extern "C" __global__ void InitBvhNodesPrimRef(
	const PrimRef* __restrict__ primitives,
	Bvh2Node* __restrict__ bvhNodes,
	u32* __restrict__ parentIdxs,
	const u32* __restrict__ primIdx,
	const u32 nLeafNodes,
	const u32 nInternalNodes
)
{
	unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (gIdx < nLeafNodes)
	{
		const u32 nodeIdx = gIdx + nInternalNodes;
		u32 idx = primIdx[gIdx];
		Bvh2Node& node = bvhNodes[nodeIdx];
		node.m_aabb.reset();
		node.m_aabb = primitives[idx].m_aabb;
		node.m_leftChildIdx = primitives[idx].m_primIdx;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		parentIdxs[nodeIdx] = INVALID_NODE_IDX;
	}
	if (gIdx < nInternalNodes)
	{
		Bvh2Node& node = bvhNodes[gIdx];
		node.m_aabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		parentIdxs[gIdx] = INVALID_NODE_IDX;
	}
}

extern "C" __global__ void BvhBuild(
	Bvh2Node* __restrict__ bvhNodes,
	u32* __restrict__ parentIdxs,
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
	parentIdxs[leftChildIdx] = gIdx;
	parentIdxs[rightChildIdx] = gIdx;
}
extern "C" __global__ void FitBvhNodes(Bvh2Node* __restrict__ bvhNodes, u32* __restrict__ parentIdxs, u32* flags, u32 nLeafNodes, u32 nInternalNodes)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= nLeafNodes) return;
	int idx = nInternalNodes + gIdx;
	u32 parent = parentIdxs[idx];
	while (atomicAdd(&flags[parent], 1) > 0)
	{
		__threadfence();
		{
			u32 leftChildIdx = bvhNodes[parent].m_leftChildIdx;
			u32 rightChildIdx = bvhNodes[parent].m_rightChildIdx;
			bvhNodes[parent].m_aabb = merge(bvhNodes[leftChildIdx].m_aabb, bvhNodes[rightChildIdx].m_aabb);
			parent = parentIdxs[parent];
			if (parent == INVALID_NODE_IDX) break;
		}
		__threadfence();
	}
}

extern "C" __global__ void CollapseToWide4Bvh(
	Bvh2Node* bvh2Nodes, 
	Bvh4Node* bvh4Nodes,
	PrimNode* bvh4LeafNodes,
	uint2* taskQ,
	u32* taskCount,
	u32* bvh8InternalNodeOffset,
	u32 nBvh2InternalNodes, 
	u32 nBvh2LeafNodes
)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	bool done = false;
	while (atomicAdd(taskCount, 0) < nBvh2LeafNodes)
	{
		__threadfence();

		if (gIdx >= nBvh2LeafNodes - 1) continue;

		uint2 task = taskQ[gIdx];
		u32 bvh2NodeIdx = task.x;
		u32 parentIdx = task.y;
		if (bvh2NodeIdx != INVALID_NODE_IDX && !done)
		{
			const Bvh2Node& node2 = bvh2Nodes[bvh2NodeIdx];
			u32 childIdx[4] = { INVALID_NODE_IDX, INVALID_NODE_IDX , INVALID_NODE_IDX , INVALID_NODE_IDX };
			Aabb childAabb[4];
			u32 childCount = 2;
			childIdx[0] = node2.m_leftChildIdx;
			childIdx[1] = node2.m_rightChildIdx;
			childAabb[0] = bvh2Nodes[node2.m_leftChildIdx].m_aabb;
			childAabb[1] = bvh2Nodes[node2.m_rightChildIdx].m_aabb;

			for (size_t j = 0; j < 2; j++) //N = 2 so we just need to expand one level to go to grandchildren
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

				Bvh2Node maxChild = bvh2Nodes[childIdx[maxAreaChildPos]];
				childIdx[maxAreaChildPos] = maxChild.m_leftChildIdx;
				childAabb[maxAreaChildPos] = bvh2Nodes[maxChild.m_leftChildIdx].m_aabb;
				childIdx[childCount] = maxChild.m_rightChildIdx;
				childAabb[childCount] = bvh2Nodes[maxChild.m_rightChildIdx].m_aabb;
				childCount++;

			}//for

			//Here we have all 4 child indices lets create wide node 
			Bvh4Node wideNode;
			wideNode.m_parent = parentIdx;
			wideNode.m_childCount = childCount;

			u32 nInternalNodes = 0;
			u32 nLeafNodes = 0;
			for (size_t i = 0; i < childCount; i++)
			{
				(childIdx[i] < nBvh2InternalNodes) ? nInternalNodes++ : nLeafNodes++;
			}

			u32 nodeOffset = atomicAdd(bvh8InternalNodeOffset, nInternalNodes);
			u32 k = 0;
			for (size_t i = 0; i < childCount; i++)
			{
				if (childIdx[i] < nBvh2InternalNodes)
				{
					wideNode.m_child[i] = nodeOffset + (k++);
					wideNode.m_aabb[i] = childAabb[i];
					taskQ[wideNode.m_child[i]] = { childIdx[i] , gIdx };
				}
				else
				{
					wideNode.m_child[i] = childIdx[i];
					bvh4LeafNodes[childIdx[i] - nBvh2InternalNodes].m_parent = gIdx;
					bvh4LeafNodes[childIdx[i] - nBvh2InternalNodes].m_primIdx = bvh2Nodes[childIdx[i]].m_leftChildIdx;
				}
			}

			atomicAdd(taskCount, nLeafNodes);
			bvh4Nodes[gIdx] = wideNode;
			done = true;
		}

		__threadfence();

		if (!__any(!done)) break;
	}
}