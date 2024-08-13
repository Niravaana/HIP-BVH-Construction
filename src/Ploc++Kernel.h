
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

extern "C" __global__ void SetupClusters(LbvhNode* bvhNodes, PrimRef* __restrict__ primRefs, u32* __restrict__ sortedPrimIdx, Aabb* __restrict__ primitivesAabb, int* __restrict__ nodeIndices, u32 primCount)
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

extern "C" __global__ void Ploc(int* nodeIndices0, int* nodeIndices1, LbvhNode* bvhNodes, PrimRef* primRefs, int* nMergedClusters, int* blockOffsetSum, int* atomicBlockCounter, u32 nClusters, u32 nInternalNodes)
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