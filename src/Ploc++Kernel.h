
#include <src/Common.h>

using namespace BvhConstruction;

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

template <typename T, typename U>
DEVICE T divideRoundUp(T value, U factor)
{
	return (value + factor - 1) / factor;
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

template <typename T>
constexpr DEVICE T Log2(T n)
{
	return n <= 1 ? 0 : 1 + Log2((n + 1) / 2);
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
	if (gIdx >= nClusters) return;

	/*if PlocRadius is 2 then we need to read 4(2 * plocRadius elements to the right and 2 * plocRadius at the end of block size, the original paper refers to as chunk.
	  Because of this neighbour research limited to chunk we dont need separate global prefix sum pass as original ploc method. This is the reason we can combine 
	  Neighbour search, merge and compact pass in one kernel.*/
	alignas(alignof(Aabb)) __shared__ u8 aabbSharedMem[sizeof(Aabb) * (PlocBlockSize + 4 * PlocRadius)];
	__shared__ u64 neighbourIndicesSharedMem[PlocBlockSize + 4 * PlocRadius];//block size of neighbouring node Indices from search of nodeIndices0
	__shared__ int nodeIndicesSharedMem[PlocBlockSize + 4 * PlocRadius]; //block size of node Indices from nodeIndices0
	__shared__ int localBlockOffset; // Each block will have a offset, use this variable to propogate localBlockOffset to all blocks

	Aabb* ptrAabbSharedMem = reinterpret_cast<Aabb*>(aabbSharedMem) + 2 * PlocRadius;
	u64* ptrNeighbourIndices  = neighbourIndicesSharedMem + 2 * PlocRadius;
	int* ptrNodeIndices = nodeIndicesSharedMem + 2 * PlocRadius;

	for (int neighbourIdx = int(threadIdx.x) - 2 * PlocRadius; neighbourIdx < int(blockDim.x) + 2 * PlocRadius; neighbourIdx += blockDim.x)
	{
		int clusterIdx = neighbourIdx + blockOffset; //global clusterIdxes
		if (clusterIdx >= 0 && clusterIdx < nClusters)
		{
			int nodeIdx = nodeIndices0[clusterIdx];
			ptrAabbSharedMem[neighbourIdx] = (nodeIdx >= nInternalNodes) ? primRefs[nodeIdx - nInternalNodes].m_aabb : bvhNodes[nodeIdx].m_aabb;
			ptrNodeIndices[neighbourIdx] = nodeIdx;
		}
		else
		{
			//Aabb x;
			//ptrAabbSharedMem[neighbourIdx] = x;
			ptrAabbSharedMem[neighbourIdx].m_min = {-FltMax, -FltMax , -FltMax };
			ptrAabbSharedMem[neighbourIdx].m_max = { FltMax, FltMax , FltMax };
			ptrNodeIndices[neighbourIdx] = INVALID_NODE_IDX;
		}
		ptrNeighbourIndices[neighbourIdx] = u64(-1);
	}

	__syncthreads();

	/*distance between i, j will be calculated in right search and similarly j, i will be calculated in left search which will be same
	  So distance is commutative. Hence we encode both in same loop by constructing 64 bit key and high 32 bits will hold area and lower 32 bits will hold i or j index.
	  Finally we store minAreaAndIndex in ptrNeighbourIndices[tId] from which we can decode back index*/
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
		int leftChildIdx = ptrNodeIndices[threadIdx.x];
		int neighbourIdx = (ptrNeighbourIndices[threadIdx.x] & 0xffffffff) - blockOffset;
		int rightChildIdx = ptrNodeIndices[neighbourIdx];
		int neighboursNeighbourIDx = (ptrNeighbourIndices[neighbourIdx] & 0xffffffff) - blockOffset;

		/* Now we will check conditions on merge
		   current cluster = threadIdx.x should be equal to neighboursNeighbourIDx i,e cluster neighbour pairs we found should be mutual.
		   For threads having smaller cluster index will only will merge ie, t1 dealing with pair (2,4) and t6 with (4,2) then t1 will do
		   the merge copy new node at cluster index 2 and t6 will mark cluster index 4 invalid */
		bool merge = false;
		if (int(threadIdx.x) == neighboursNeighbourIDx)
		{
			/*(Point A) For our example t1 and t6 will come here but t6 wont go in if so do nothing
			So for t6 the nodeIdx will be invalid value*/
			if (int(threadIdx.x) < neighbourIdx) merge = true;
		}
		else
		{
			//Rest of the threads will just copy cluster index they are processing as is
			nodeIdx = leftChildIdx;
		}

		/*
		  Every thread will have merge valur to 1 or 0. So if we have 6 threads with following merge values 
		  [ 0 1 0 1 1 0 0] -> implies 3 threads will merge and will have new nodes.
		  So we can efficiently calculate binary prefix sum using binaryWarpPrefixSum.
		  For a given thread it will return prefix sum value which is new node offset for that thread.
		  in above example for thread t3 binaryWarpPrefixSum will return 2 and nMergedClusters will return 
		  3 which is total merged nodes this block produced. This value will be useful to calculate new nClusters 
		  after this kernel call is done.
		*/
		int mergedNodeIdx = nClusters - 2 - binaryWarpPrefixSum(merge, nMergedClusters);
		if (merge)
		{
			Aabb aabb = ptrAabbSharedMem[threadIdx.x];
			aabb.grow(ptrAabbSharedMem[neighbourIdx]);
			bvhNodes[mergedNodeIdx].m_leftChildIdx = leftChildIdx;
			bvhNodes[mergedNodeIdx].m_rightChildIdx = rightChildIdx;
			bvhNodes[mergedNodeIdx].m_aabb = aabb;
			nodeIdx = mergedNodeIdx;
		}
	}

	__syncthreads();

	/*
	  Now we need to do compaction pass if you notice (Point A)
	  After merging pass for all the threads that had found valid neighbour cluster pairs but did not merge the nodeIdx will have invalid value.
	  In compaction phase we want to shift all the invalid values at the end of nodeIndices array.
	  We calculate nodes that dont have invalid idx. Again we use binary prefix sum on condition nodeIdx != invalid idx but for the block.
	*/

	int newblockOffset = binaryBlockPrefixSum(nodeIdx != INVALID_NODE_IDX, nodeIndicesSharedMem);
	
	/*
	  We want to find new positions(cluster Indices) in nodeIndices array where we want to copy merged node and the nodes which are still remaining to be processed(which did not found any neighbours yet).
	  Both of these nodeIdx will have not invalid value set to them. So in given block we need to calculate how many much nodes are there.
	  Though blockOffset is for current block say block0,  we need to add it to blockOffset of block1 then blockOffset of block0 + block1 to block2 so on. 
	  We do this below using one thread in block to calculate blockOffset and it waits on the atomic counter that is set by the previous block thread.
	*/

	if (threadIdx.x == blockDim.x - 1)
	{
		while (atomicAdd(atomicBlockCounter, 0) < blockIdx.x)
			; //wait for prev block thread to increament counter
		localBlockOffset = atomicAdd(blockOffsetSum, newblockOffset); // If we are block3 then get sum of blockOffset0 + blockOffset1 + blockOffset2
		atomicAdd(atomicBlockCounter, 1);
	}

	__syncthreads();

	/*
	  Put merged nodeIdx into the new cluster positions we calculated above by compaction.
	*/
	if (gIdx < nClusters)
	{
		if (nodeIdx != INVALID_NODE_IDX)
		{
			int newClusterIdx = localBlockOffset + newblockOffset - 1;
			nodeIndices1[newClusterIdx] = nodeIdx; //We copy node indices after compaction other buffer
		}
	}
}