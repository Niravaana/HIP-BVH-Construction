
#include <src/Common.h>

using namespace BvhConstruction;

template <typename T>
constexpr DEVICE T Log2(T n)
{
	return n <= 1 ? 0 : 1 + Log2((n + 1) / 2);
}

template <typename T, typename U>
DEVICE T divideRoundUp(T value, U factor)
{
	return (value + factor - 1) / factor;
}

extern "C" __global__ void SetupClusters(Bvh2Node* bvhNodes, PrimRef* __restrict__ primRefs, u32* __restrict__ sortedPrimIdx, Aabb* __restrict__ primitivesAabb, int* __restrict__ nodeIndices, u32 primCount)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= primCount) return;

	u32 primIdx = sortedPrimIdx[gIdx];
	primRefs[gIdx].m_primIdx = primIdx;
	primRefs[gIdx].m_aabb = primitivesAabb[primIdx];
	nodeIndices[gIdx] = gIdx + (primCount - 1);

	if (gIdx < (primCount - 1))
	{
		bvhNodes[gIdx].m_leftChildIdx = INVALID_NODE_IDX;
		bvhNodes[gIdx].m_rightChildIdx = INVALID_NODE_IDX;
		bvhNodes[gIdx].m_aabb.reset();
	}
}

DEVICE int binaryWarpPrefixSum(bool warpVal, int* counter)
{
	const int	   laneIndex = threadIdx.x & (WarpSize - 1);
	const uint64_t warpBallot = __ballot(warpVal);
	const int	   warpCount = __popcll(warpBallot);
	const int	   warpSum = __popcll(warpBallot & ((1ull << laneIndex) - 1ull));
	int			   warpOffset;
	if (laneIndex == __ffsll(static_cast<unsigned long long>(warpBallot)) - 1)
		warpOffset = atomicAdd(counter, warpCount);
	warpOffset = __shfl(warpOffset, __ffsll(static_cast<unsigned long long>(warpBallot)) - 1);
	return warpOffset + warpSum;
}

DEVICE int binaryBlockPrefixSum(bool blockVal, int* blockCache)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	const int warpIndex = threadIdx.x >> Log2(WarpSize);
	const int warpsPerBlock = divideRoundUp(static_cast<int>(blockDim.x), WarpSize);

	int			   blockValue = blockVal;
	const uint64_t warpBallot = __ballot(blockVal);
	const int	   warpCount = __popcll(warpBallot);
	const int	   warpSum = __popcll(warpBallot & ((1ull << laneIndex) - 1ull));

	if (laneIndex == 0) blockCache[warpIndex] = warpCount;

	__syncthreads();
	if (threadIdx.x < warpsPerBlock) blockValue = blockCache[threadIdx.x];

	for (int i = 1; i < warpsPerBlock; i <<= 1)
	{
		__syncthreads();
		if (threadIdx.x < warpsPerBlock && threadIdx.x >= i) blockValue += blockCache[threadIdx.x - i];
		__syncthreads();
		if (threadIdx.x < warpsPerBlock) blockCache[threadIdx.x] = blockValue;
	}

	__syncthreads();
	return blockCache[warpIndex] + warpSum - warpCount + blockVal;
}

extern "C" __global__ void SinglePassPloc(int * nodeIndices, Bvh2Node* bvhNodes, PrimRef* primRefs, u32 nClusters, u32 nInternalNodes)
{
	int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x > 0) return;
	if (gIdx > PlocBlockSize) return;

	__shared__ int nodeIndicesSharedMem[PlocBlockSize]; 
	alignas(alignof(Aabb)) __shared__ u8 aabbCache[sizeof(Aabb) * PlocBlockSize];
	__shared__ u64 neighbourIndicesSharedMem[PlocBlockSize];
	__shared__ int nNewClusters;
	__shared__ int nMergedClusters;
	Aabb* aabbSharedMem = reinterpret_cast<Aabb*>(aabbCache);

	if (gIdx >= 0 && gIdx < nClusters)
	{
		int nodeIdx = nodeIndices[gIdx];
		aabbSharedMem[gIdx] = (nodeIdx >= nInternalNodes) ? primRefs[nodeIdx - nInternalNodes].m_aabb : bvhNodes[nodeIdx].m_aabb;
		nodeIndicesSharedMem[gIdx] = nodeIdx;
	}
	else
	{
		aabbSharedMem[gIdx].m_min = { -FltMax, -FltMax , -FltMax };
		aabbSharedMem[gIdx].m_max = { FltMax, FltMax , FltMax };
		nodeIndicesSharedMem[gIdx] = INVALID_NODE_IDX;
	}
	__syncthreads();

	while (nClusters > 1)
	{
		int nodeIdx = nodeIndicesSharedMem[gIdx];
		neighbourIndicesSharedMem[gIdx] = (u64)-1;
		__syncthreads();

		u64 minAreadAndIndex = u64(-1);
		Aabb aabb = aabbSharedMem[gIdx];

		for (int neighbourIdx = gIdx + 1; neighbourIdx < min(nClusters, gIdx + PlocRadius + 1); neighbourIdx++)
		{
			Aabb neighbourAabb = aabbSharedMem[neighbourIdx];
			neighbourAabb.grow(aabb);
			float area = neighbourAabb.area();

			u64 encode0 = (u64(__float_as_int(area)) << 32ull) | u64(neighbourIdx);
			minAreadAndIndex = min(minAreadAndIndex, encode0);
			u64 encode1 = (u64(__float_as_int(area)) << 32ull) | u64(gIdx);
			atomicMin(&neighbourIndicesSharedMem[neighbourIdx], encode1);
		}

		atomicMin(&neighbourIndicesSharedMem[gIdx], minAreadAndIndex);

		if (gIdx == 0) nMergedClusters = 0;

		__syncthreads();
		

		if (gIdx < nClusters)
		{
			int currentClusterIdx = gIdx;
			int leftChildIdx = nodeIndicesSharedMem[currentClusterIdx];
			int neighbourIdx = (neighbourIndicesSharedMem[currentClusterIdx] & 0xffffffff);
			int rightChildIdx = nodeIndicesSharedMem[neighbourIdx];
			int neighboursNeighbourIDx = (neighbourIndicesSharedMem[neighbourIdx] & 0xffffffff);

			bool merge = false;
			if (currentClusterIdx == neighboursNeighbourIDx)
			{
				if (currentClusterIdx < neighbourIdx) 
					merge = true;
				else
				{
					aabb.m_min = { -FltMax, -FltMax , -FltMax };
					aabb.m_max = { FltMax, FltMax , FltMax };
					nodeIdx = INVALID_NODE_IDX;
				}
			}

			int mergedNodeIdx = nClusters - 2 - binaryWarpPrefixSum(merge, &nMergedClusters);
			if (merge)
			{
				aabb.grow(aabbSharedMem[neighbourIdx]);
				bvhNodes[mergedNodeIdx].m_leftChildIdx = leftChildIdx;
				bvhNodes[mergedNodeIdx].m_rightChildIdx = rightChildIdx;
				bvhNodes[mergedNodeIdx].m_aabb = aabb;
				nodeIdx = mergedNodeIdx;
			}
		}

		__syncthreads();

		int newblockOffset = binaryBlockPrefixSum(nodeIdx != INVALID_NODE_IDX, nodeIndicesSharedMem);
		aabbSharedMem[gIdx].m_min = { -FltMax, -FltMax , -FltMax };
		aabbSharedMem[gIdx].m_max = { FltMax, FltMax , FltMax };
		nodeIndicesSharedMem[gIdx] = INVALID_NODE_IDX;
		
		__syncthreads();

		if (gIdx == blockDim.x - 1) nNewClusters = newblockOffset;

		if (gIdx < nClusters)
		{
			if (nodeIdx != INVALID_NODE_IDX)
			{
				const int newClusterIdx = newblockOffset - 1;
				aabbSharedMem[newClusterIdx] = aabb;
				nodeIndicesSharedMem[newClusterIdx] = nodeIdx;
			}
		}

		__syncthreads();
		nClusters = nNewClusters;
	}//while
}

extern "C" __global__ void Ploc(int* nodeIndices0, int* nodeIndices1, Bvh2Node* bvhNodes, PrimRef* primRefs, int* nMergedClusters, int* blockOffsetSum, int* atomicBlockCounter, u32 nClusters, u32 nInternalNodes)
{
	int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int blockOffset = blockDim.x * blockIdx.x;

	alignas(alignof(Aabb)) __shared__ u8 aabbSharedMem[sizeof(Aabb) * (PlocBlockSize + 4 * PlocRadius)];
	__shared__ u64 neighbourIndicesSharedMem[PlocBlockSize + 4 * PlocRadius];//block size of neighbouring node Indices from search of nodeIndices0
	__shared__ int nodeIndicesSharedMem[PlocBlockSize + 4 * PlocRadius]; //block size of node Indices from nodeIndices0
	__shared__ int localBlockOffset; // Each block will have a offset, use this variable to propogate localBlockOffset to all blocks

	Aabb* ptrAabbSharedMem = reinterpret_cast<Aabb*>(aabbSharedMem) + 2 * PlocRadius;
	u64* ptrNeighbourIndices  = neighbourIndicesSharedMem + 2 * PlocRadius;
	int* ptrNodeIndices = nodeIndicesSharedMem + 2 * PlocRadius;

	for (int neighbourIdx = int(threadIdx.x) - 2 * PlocRadius; neighbourIdx < int(blockDim.x) + 2 * PlocRadius; neighbourIdx += blockDim.x)
	{
		int clusterIdx = neighbourIdx + blockOffset; 
		if (clusterIdx >= 0 && clusterIdx < nClusters)
		{
			int nodeIdx = nodeIndices0[clusterIdx];
			ptrAabbSharedMem[neighbourIdx] = (nodeIdx >= nInternalNodes) ? primRefs[nodeIdx - nInternalNodes].m_aabb : bvhNodes[nodeIdx].m_aabb;
			ptrNodeIndices[neighbourIdx] = nodeIdx;
		}
		else
		{
			ptrAabbSharedMem[neighbourIdx].m_min = {-FltMax, -FltMax , -FltMax };
			ptrAabbSharedMem[neighbourIdx].m_max = { FltMax, FltMax , FltMax };
			ptrNodeIndices[neighbourIdx] = INVALID_NODE_IDX;
		}
		ptrNeighbourIndices[neighbourIdx] = u64(-1);
	}

	__syncthreads();

	for (int tId = int(threadIdx.x) - 2 * PlocRadius; tId < int(blockDim.x) + PlocRadius; tId += blockDim.x)
	{
		u64 minAreadAndIndex = u64(-1);
		Aabb aabb = ptrAabbSharedMem[tId];

		for (int neighbourIdx = tId + 1; neighbourIdx < tId + PlocRadius + 1; neighbourIdx++)
		{
			Aabb neighbourAabb = ptrAabbSharedMem[neighbourIdx];
			neighbourAabb.grow(aabb);
			float area = neighbourAabb.area();

			u64 encode0 = (u64(__float_as_int(area)) << 32ull) | u64(neighbourIdx + blockOffset);
			minAreadAndIndex = min(minAreadAndIndex, encode0);
			u64 encode1 = (u64(__float_as_int(area)) << 32ull) | u64(tId + blockOffset);
			atomicMin(&ptrNeighbourIndices[neighbourIdx], encode1);
		}
	
		 atomicMin(&ptrNeighbourIndices[tId], minAreadAndIndex);
	}

	__syncthreads();
	
	int nodeIdx = INVALID_NODE_IDX;

	if (gIdx < nClusters)
	{
		int currentClusterIdx = threadIdx.x;
		int leftChildIdx = ptrNodeIndices[currentClusterIdx];
		int neighbourIdx = (ptrNeighbourIndices[currentClusterIdx] & 0xffffffff) - blockOffset;
		int rightChildIdx = ptrNodeIndices[neighbourIdx];
		int neighboursNeighbourIDx = (ptrNeighbourIndices[neighbourIdx] & 0xffffffff) - blockOffset;
		
		bool merge = false;
		if (currentClusterIdx == neighboursNeighbourIDx)
		{
			if (currentClusterIdx < neighbourIdx) merge = true;
		}
		else
		{
			nodeIdx = leftChildIdx;
		}
		
		int mergedNodeIdx = nClusters - 2 - binaryWarpPrefixSum(merge, nMergedClusters);
		if (merge)
		{
			Aabb aabb = ptrAabbSharedMem[currentClusterIdx];
			aabb.grow(ptrAabbSharedMem[neighbourIdx]);
			bvhNodes[mergedNodeIdx].m_leftChildIdx = leftChildIdx;
			bvhNodes[mergedNodeIdx].m_rightChildIdx = rightChildIdx;
			bvhNodes[mergedNodeIdx].m_aabb = aabb;
			nodeIdx = mergedNodeIdx;
		}
	
	}

	__syncthreads();

	int newblockOffset = binaryBlockPrefixSum(nodeIdx != INVALID_NODE_IDX, nodeIndicesSharedMem);
	
	if (threadIdx.x == blockDim.x - 1)
	{
		
		while (atomicAdd(atomicBlockCounter, 0) < blockIdx.x);
		localBlockOffset = atomicAdd(blockOffsetSum, newblockOffset);
		atomicAdd(atomicBlockCounter, 1);
	}

	__syncthreads();

	if (gIdx < nClusters)
	{
		if (nodeIdx != INVALID_NODE_IDX)
		{
			int newClusterIdx = localBlockOffset + newblockOffset - 1;
			nodeIndices1[newClusterIdx] = nodeIdx ; 
		}
	}
}

extern "C" __global__ void CollapseToWide4Bvh(
	Bvh2Node* bvh2Nodes, 
	PrimRef* bvh2LeafNodes, 
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
					bvh4LeafNodes[childIdx[i] - nBvh2InternalNodes].m_primIdx = bvh2LeafNodes[childIdx[i] - nBvh2InternalNodes].m_primIdx;
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