
#include <src/Common.h>

using namespace BvhConstruction;

extern "C" __global__ void SetupClusters(PrimRef* __restrict__ primRefs, u32* __restrict__ sortedPrimIdx, Aabb* __restrict__ primitivesAabb, u32* __restrict__ nodeIndices, u32 primCount)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= primCount) return;

	u32 primIdx = sortedPrimIdx[gIdx];
	primRefs[gIdx].m_primIdx = primIdx;
	primRefs[gIdx].m_aabb = primitivesAabb[primIdx];
	nodeIndices[gIdx] = gIdx + (primCount - 1);
}

extern "C" __global__ void Ploc(u32* nodeIndices0, u32* nodeIndices1, LbvhNode* bvhNodes, PrimRef* primRefs, u32* nMergedClusters, u32 nClusters, u32 nInternalNodes)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	u32 blockOffset = blockDim.x * blockIdx.x;
	if (gIdx >= nClusters) return;

	/*if PlocRadius is 2 then we need to read 4(2 * plocRadius elements to the right and 2 * plocRadius at the end of block size, the original paper refers to as chunk.
	  Because of this neighbour research limited to chunk we dont need separate global prefix sum pass as original ploc method. This is the reason we can combine 
	  Neighbour search, merge and compact pass in one kernel.*/
	__shared__ u8 aabbSharedMem[sizeof(Aabb) * PlocBlockSize + 4 * PlocRadius];
	__shared__ u64 neighbourIndicesSharedMem[PlocBlockSize + 4 * PlocRadius];//block size of neighbouring node Indices from search of nodeIndices0
	__shared__ u32 nodeIndicesSharedMem[PlocBlockSize + 4 * PlocRadius]; //block size of node Indices from nodeIndices0
	__shared__ int localBlockOffset; // Each block will have a offset, use this variable to propogate localBlockOffset to all blocks

	Aabb* ptrAabbSharedMem = reinterpret_cast<Aabb*>(aabbSharedMem + 2 * PlocRadius);
	u64* ptrNeighbourIndices  = neighbourIndicesSharedMem + 2 * PlocRadius;
	u32* ptrNodeIndices = nodeIndicesSharedMem + 2 * PlocRadius;

	for (int neighbourIdx = threadIdx.x - 2 * PlocRadius; neighbourIdx < blockDim.x + 2 * PlocRadius; neighbourIdx += blockDim.x)
	{
		int clusterIdx = neighbourIdx + blockOffset; //global clusterIdxes
		if (clusterIdx >= 0 && clusterIdx < nClusters)
		{
			u32 nodeIdx = nodeIndices0[clusterIdx];
			ptrAabbSharedMem[neighbourIdx] = (nodeIdx >= nInternalNodes) ? primRefs[nodeIdx - nInternalNodes].m_aabb : bvhNodes[nodeIdx].m_aabb;
			ptrNodeIndices[neighbourIdx] = nodeIdx;
		}
		else
		{
			ptrAabbSharedMem[neighbourIdx].reset();
			ptrNodeIndices[neighbourIdx] = INVALID_NODE_IDX;
		}
		ptrNeighbourIndices[neighbourIdx] = u64(-1);
	}
}