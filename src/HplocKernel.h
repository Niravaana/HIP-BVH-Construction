
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

extern "C" __global__ void SetupClusters(Bvh2Node* bvhNodes, PrimRef* __restrict__ primRefs, u32* __restrict__ sortedPrimIdx, Aabb* __restrict__ primitivesAabb, int* __restrict__ nodeIndices, u32* __restrict__ parentIdx, u32 primCount)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= primCount) return;

	u32 primIdx = sortedPrimIdx[gIdx];
	primRefs[gIdx].m_primIdx = primIdx;
	primRefs[gIdx].m_aabb = primitivesAabb[primIdx];
	nodeIndices[gIdx] = gIdx + (primCount - 1);
	parentIdx[gIdx] = INVALID_NODE_IDX;

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

DEVICE uint64_t findHighestDiffBit(const u32* __restrict__ mortonCodes, int i, int j, int n)
{
	if (j < 0 || j >= n) return ~0ull;
	const uint64_t a = (static_cast<uint64_t>(mortonCodes[i]) << 32ull) | i;
	const uint64_t b = (static_cast<uint64_t>(mortonCodes[j]) << 32ull) | j;
	return a ^ b;
}

DEVICE int findParent(
	const u32* __restrict__ mortonCodes,
	int i,
	int j,
	int n)
{
	if (i == 0 && j == n) return INVALID_NODE_IDX;
	if (i == 0 || (j != n && findHighestDiffBit(mortonCodes, j - 1, j, n) < findHighestDiffBit(mortonCodes, i - 1, i, n)))
	{
		return j - 1;
	}
	else
	{
		return i - 1;
	}
}

DEVICE void findNearestNeighbours(u32 nPrims, u64* nearestNeighbours, int* clusterIndices, Aabb* aabbSharedMem, Bvh2Node* bvhNodes, PrimRef* primRefs, u32 nInternalNodes)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	u32 nodeIdx = clusterIndices[laneIndex];
	Aabb aabb = aabbSharedMem[laneIndex];
	u64 minAreadAndIndex = u64(-1);

	for (int neighbourIdx = laneIndex + 1; neighbourIdx < min(nPrims, laneIndex + PlocRadius + 1); neighbourIdx++)
	{
		Aabb neighbourAabb = aabbSharedMem[neighbourIdx];
		neighbourAabb.grow(aabb);
		float area = neighbourAabb.area();

		u64 encode0 = (u64(__float_as_int(area)) << 32ull) | u64(neighbourIdx);
		minAreadAndIndex = min(minAreadAndIndex, encode0);
		u64 encode1 = (u64(__float_as_int(area)) << 32ull) | u64(laneIndex);
		atomicMin(&nearestNeighbours[neighbourIdx], encode1);
	}

	atomicMin(&nearestNeighbours[laneIndex], minAreadAndIndex);
	__syncthreads();
}

DEVICE int mergeClusters(u32 nPrims, u64* nearestNeighbours, int* clusterIndices, Aabb* aabbSharedMem, Bvh2Node* bvhNodes, PrimRef* primRefs, int* nMergedClusters, u32 nInternalNodes)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	
	int currentClusterIdx = laneIndex;
	int leftChildIdx = clusterIndices[currentClusterIdx];
	int neighbourIdx = (nearestNeighbours[currentClusterIdx] & 0xffffffff);
	int rightChildIdx = clusterIndices[neighbourIdx];
	int neighboursNeighbourIDx = (nearestNeighbours[neighbourIdx] & 0xffffffff);

	int nodeIdx = INVALID_NODE_IDX;
	bool merge = false;
	if (currentClusterIdx == neighboursNeighbourIDx)
	{
		if (currentClusterIdx < neighbourIdx)
			merge = true;
	}
	else
	{
		nodeIdx = leftChildIdx;
	}

	Aabb aabb = aabbSharedMem[currentClusterIdx];
	int mergedNodeIdx = nInternalNodes - binaryWarpPrefixSum(merge, nMergedClusters);
	if (merge)
	{
		aabb.grow(aabbSharedMem[neighbourIdx]);
		bvhNodes[mergedNodeIdx].m_leftChildIdx = leftChildIdx;
		bvhNodes[mergedNodeIdx].m_rightChildIdx = rightChildIdx;
		bvhNodes[mergedNodeIdx].m_aabb = aabb;
		nodeIdx = mergedNodeIdx;
	}

	__syncthreads();

	clusterIndices[currentClusterIdx] = nodeIdx;
	aabbSharedMem[currentClusterIdx] = aabb;

	__syncthreads();

	return (nPrims - (*nMergedClusters));
}

DEVICE u32 loadIndices(u32 start, u32 end, int* nodeIndices, int* clusterIndices, Aabb* aabbSharedMem, Bvh2Node* bvhNodes, PrimRef* primRefs, u32 nInternalNodes, u32 offset)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	u32 nClusters = end - start;
	int nodeOffset = start + laneIndex;
	bool isLaneActive = (laneIndex) < nClusters;

	if (isLaneActive)
	{
		int nodeIdx = nodeIndices[nodeOffset];
		clusterIndices[laneIndex + offset] = nodeIdx;
		aabbSharedMem[laneIndex + offset] = (nodeIdx >= nInternalNodes) ? primRefs[nodeIdx - nInternalNodes].m_aabb : bvhNodes[nodeIdx].m_aabb;
	}
	__syncthreads();

	return __popcll(__ballot(isLaneActive));
}

DEVICE void storeIndices(u32 nClusters, int* clusterIndices, int* nodeIndices0, u32 Lstart)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	bool isLaneActive = (laneIndex) < nClusters;

	if (isLaneActive)
	{
		if (clusterIndices[laneIndex] != INVALID_NODE_IDX)
		{
			nodeIndices0[Lstart + laneIndex] = clusterIndices[laneIndex];
		}
	}
}

DEVICE void plocMerge(u32 laneId, u32 L, u32 R, u32 split, bool finalR, int* nodeIndices0, Bvh2Node* bvhNodes, PrimRef* primRefs, int* nMergedClusters, u32 nInternalNodes)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	alignas(alignof(Aabb)) __shared__ u8 aabbCache[sizeof(Aabb) * WarpSize];
	__shared__ int clusterIndices[WarpSize];
	__shared__ u64 nearestNeighbours[WarpSize];
	Aabb* aabbSharedMem = reinterpret_cast<Aabb*>(aabbCache);

	u32 Lstart = __shfl(L, laneId);
	u32 Lend = __shfl(split, laneId);
	u32 Rstart = __shfl(split, laneId);
	u32 Rend = __shfl(R, laneId) + 1;

	nearestNeighbours[laneIndex] = (u64)-1;
	clusterIndices[laneIndex] = INVALID_NODE_IDX;
	__syncthreads();

	u32 nLeft = loadIndices(Lstart, Lend, nodeIndices0, clusterIndices, aabbSharedMem, bvhNodes, primRefs, nInternalNodes, 0);
	u32 nRight = loadIndices(Rstart, Rend, nodeIndices0, clusterIndices, aabbSharedMem, bvhNodes, primRefs, nInternalNodes, nLeft);
	u32 nPrims = nLeft + nRight;

	u32 threshold = (__shfl(finalR, laneIndex) == true) ? 1 : (WarpSize / 2);

	while (nPrims > threshold)
	{
		findNearestNeighbours(nPrims, nearestNeighbours, clusterIndices, aabbSharedMem, bvhNodes, primRefs, nInternalNodes);
		nPrims = mergeClusters(nPrims, nearestNeighbours, clusterIndices, aabbSharedMem, bvhNodes, primRefs, nMergedClusters,nInternalNodes);
	}

	storeIndices(nLeft + nRight, clusterIndices, nodeIndices0, Lstart);
}

extern "C" __global__ void HPloc(Bvh2Node* bvhNodes, PrimRef* primRefs, u32* mortonCodes, int* nodeIndices0, u32* parentIdx, int* nMergedClusters, u32 nClusters, u32 nInternalNodes)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	u32 L = gIdx;
	u32 R = gIdx;
	bool laneIsActive = (gIdx < nClusters);
	u32 previousId = INVALID_NODE_IDX;
	u32 split = INVALID_VALUE;

	while (__ballot(laneIsActive))
	{
		if (laneIsActive)
		{
			if (findParent(mortonCodes, L, R, nClusters) == R)
			{
				previousId = atomicExch(&parentIdx[R], L);
				if (previousId != INVALID_NODE_IDX)
				{
					split = R + 1;
					R = previousId;
				}
			}
			else
			{
				previousId = atomicExch(&parentIdx[L-1], R);
				if (previousId != INVALID_NODE_IDX)
				{
					split = L;
					L = previousId;
				}
			}

			if (previousId == INVALID_NODE_IDX)
			{
				laneIsActive = false;
			}
		}//if laneIsActive end

		u32 size = R - L + 1;
		bool finalR = ((laneIsActive) && (size == nClusters)); //reached root need to stop
		u64 waveMask = __ballot((laneIsActive) && ((size > WarpSize / 2) || finalR));
		
		while (waveMask)
		{
			u32 laneId = __ffsll(static_cast<unsigned long long>(waveMask));
			//plocMerge(laneId, L, R, split, finalR, nodeIndices0, bvhNodes, primRefs, nMergedClusters, nInternalNodes);
			waveMask = waveMask & (waveMask - 1);
		}//end while 
	}//While ballot(laneIsActive) end
}