#include <src/Common.h>

using namespace BvhConstruction;

constexpr u32 ExtentCacheSize = sizeof(Aabb) * MaxBatchedBlockSize;
constexpr u32 MortonKeyCacheSize = sizeof(u32) * MaxBatchedBlockSize;
constexpr u32 MortonValueCacheSize = sizeof(u32) * MaxBatchedBlockSize;
constexpr u32 CounterBufferCacheSize = sizeof(u32) * MaxBatchedBlockSize;
constexpr u32 BatchBuilderCacheSize = ExtentCacheSize + MortonKeyCacheSize + MortonValueCacheSize + CounterBufferCacheSize;

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

//ToDo move these common functions inside utilityKernels file
DEVICE INLINE Aabb shflAabb(const Aabb& aabb, u32 srcLane)
{
	Aabb b;
	b.m_min.x = __shfl(aabb.m_min.x, srcLane);
	b.m_min.y = __shfl(aabb.m_min.y, srcLane);
	b.m_min.z = __shfl(aabb.m_min.z, srcLane);
	b.m_max.x = __shfl(aabb.m_max.x, srcLane);
	b.m_max.y = __shfl(aabb.m_max.y, srcLane);
	b.m_max.z = __shfl(aabb.m_max.z, srcLane);
	return b;
}

DEVICE INLINE Aabb warpReduce(Aabb val)
{
	const int warpIndex = threadIdx.x & (WarpSize - 1);
	Aabb warpValue = shflAabb(val, warpIndex ^ 1);
	val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	warpValue = shflAabb(val, warpIndex ^ 2);
	val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	warpValue = shflAabb(val, warpIndex ^ 4);
	val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	warpValue = shflAabb(val, warpIndex ^ 8);
	val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	warpValue = shflAabb(val, warpIndex ^ 16);
	val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	if constexpr (WarpSize == 64)
	{
		warpValue = shflAabb(val, warpIndex ^ 32);
		val = { min(val.m_min, warpValue.m_min), max(val.m_max, warpValue.m_max) };
	}
	val = shflAabb(val, WarpSize - 1);
	return val;
}

DEVICE INLINE Aabb blockReduce(Aabb val, __shared__ Aabb* sharedMem)
{
	Aabb nullAabb;
	int laneId = threadIdx.x % WarpSize;
	int warpId = threadIdx.x / WarpSize;

	val = warpReduce(val);

	if (laneId == 0) sharedMem[warpId] = val;

	__syncthreads();

	val = (threadIdx.x < blockDim.x / WarpSize) ? sharedMem[laneId] : nullAabb;

	if (warpId == 0) val = warpReduce(val);

	return val;
}

DEVICE u32 morton3D(u32 x)
{
	x = (x * 0x00010001u) & 0xFF0000FFu;
	x = (x * 0x00000101u) & 0x0F00F00Fu;
	x = (x * 0x00000011u) & 0xC30C30C3u;
	x = (x * 0x00000005u) & 0x49249249u;
	return x;
}

DEVICE u32 computeMortonCode(float3 normalizedPos, float3 sceneExtent)
{
	float	 x = min(max(normalizedPos.x * 1024.0f, 0.0f), 1023.0f);
	float	 y = min(max(normalizedPos.y * 1024.0f, 0.0f), 1023.0f);
	float	 z = min(max(normalizedPos.z * 1024.0f, 0.0f), 1023.0f);

	u32 xx = morton3D(u32(x));
	u32 yy = morton3D(u32(y));
	u32 zz = morton3D(u32(z));

	return (xx * 4 + yy * 2 + zz);
}

DEVICE int binaryBlockPrefixSum(bool blockVal, u32* blockCache)
{
	const int laneIndex = threadIdx.x & (WarpSize - 1);
	const int warpIndex = threadIdx.x >> Log2(WarpSize);
	const int warpsPerBlock = divideRoundUp(static_cast<int>(blockDim.x), WarpSize);

	u32			   blockValue = blockVal;
	const uint64_t warpBallot = __ballot(blockVal);
	const u32	   warpCount = __popcll(warpBallot);
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
	if (i < 0 || j >= n) return ~0ull;
	const uint64_t a = (static_cast<uint64_t>(mortonCodes[i]) << 32ull) | i;
	const uint64_t b = (static_cast<uint64_t>(mortonCodes[j]) << 32ull) | j;
	return a ^ b;
}

DEVICE int findParent(
	Bvh2Node* __restrict__ bvhNodes,
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

DEVICE void BvhBuildAndFit(
	u32 primIdx,
	Bvh2Node* bvhNodes,
	PrimRef* bvhLeafNodes,
	u32*  bvhNodeCounter,
	uint2* spans,
	const u32* mortonCodes,
	u32 nLeafNodes,
	u32 nInternalNodes)
{
	if (primIdx >= nLeafNodes) return;

	int i = primIdx;
	int j = primIdx + 1;

	int parentIdx = findParent(bvhNodes, spans, mortonCodes, nInternalNodes + primIdx, i, j, nLeafNodes);
	primIdx = parentIdx;

	while (atomicAdd(&bvhNodeCounter[primIdx], 1) > 0)
	{
		__threadfence();

		Bvh2Node& node = bvhNodes[primIdx];
		uint2 span = spans[primIdx];

		Aabb leftAabb = (node.m_leftChildIdx >= nInternalNodes) ? bvhLeafNodes[node.m_leftChildIdx - nInternalNodes].m_aabb : bvhNodes[node.m_leftChildIdx].m_aabb;
		Aabb rightAabb = (node.m_rightChildIdx >= nInternalNodes) ? bvhLeafNodes[node.m_rightChildIdx - nInternalNodes].m_aabb : bvhNodes[node.m_rightChildIdx].m_aabb;
		node.m_aabb = merge(leftAabb, rightAabb);

		parentIdx = findParent(bvhNodes, spans, mortonCodes, primIdx, span.x, span.y, nLeafNodes);

		if (parentIdx == INVALID_NODE_IDX)
		{
			bvhNodeCounter[nLeafNodes - 1] = primIdx; //Saving root node;
			break;
		}

		primIdx = parentIdx;

		__threadfence();
	}
}

extern "C" __global__ void BatchedBuildKernelLbvh(
	D_BatchedBuildInputs* __restrict__ batchedBuildInputs,
	Bvh2Node* __restrict__ bvhNodes,
	PrimRef* __restrict__ bvhLeafNodes,
	uint2* __restrict__ spans,
	u32* __restrict__ rootNodes,
	const u32 nBatches,
	Aabb* d_sceneExtent)
{
	const u32 gIdx = blockIdx.x + gridDim.x * blockIdx.y;
	if (gIdx >= nBatches) return;

	rootNodes[gIdx] = INVALID_NODE_IDX;
	D_BatchedBuildInputs batchedBuildInput = batchedBuildInputs[gIdx];

	const u32 primIdx = threadIdx.x;
	const u32 primCount = batchedBuildInput.m_nPrimtives;
	const u32 bvhLeafNodeOffset = gIdx * primCount;
	const u32 bvhNodeOffset = gIdx * (primCount - 1);
	
	//Calculate scene extents
	Aabb primAabb;

	if (primIdx < primCount)
	{
		primAabb.grow(batchedBuildInput.m_prims[primIdx].v1);
		primAabb.grow(batchedBuildInput.m_prims[primIdx].v2);
		primAabb.grow(batchedBuildInput.m_prims[primIdx].v3);
		
		bvhLeafNodes[bvhLeafNodeOffset + primIdx].m_aabb = primAabb;
		bvhLeafNodes[bvhLeafNodeOffset + primIdx].m_primIdx = primIdx;
	}
	else
	{
		primAabb.grow(batchedBuildInput.m_prims[primCount - 1].v1);
		primAabb.grow(batchedBuildInput.m_prims[primCount - 1].v2);
		primAabb.grow(batchedBuildInput.m_prims[primCount - 1].v3);
	}

	__shared__ u8 aabbSharedMem[ExtentCacheSize];
	Aabb* ptrAabbMem = reinterpret_cast<Aabb*>(aabbSharedMem);
	Aabb sceneExtent = blockReduce(primAabb, ptrAabbMem);
	//ToDo remove debug
	d_sceneExtent[gIdx] = sceneExtent;
	
	__syncthreads();

	//Caluclate morton codes 
	const u32 warpsPerBlock = divideRoundUp(static_cast<u32>(blockDim.x), WarpSize);
	__shared__ u32 mortonCodeKeys[MaxBatchedBlockSize];
	__shared__ u32 mortonCodeValues[MaxBatchedBlockSize];
	__shared__ u32 bvhNodeCounter[MaxBatchedBlockSize];
	bvhNodeCounter[primIdx] = 0;

	if (primIdx < primCount)
	{
		float3 centre = primAabb.center();
		float3 extents = sceneExtent.extent();
		float3 normalizedCentroid = (centre - sceneExtent.m_min) / extents;

		mortonCodeKeys[primIdx] = computeMortonCode(normalizedCentroid, extents);
		mortonCodeValues[primIdx] = primIdx;
	}
	else
	{
		mortonCodeKeys[primIdx] = INVALID_VALUE;
		mortonCodeValues[primIdx] = INVALID_VALUE;
	}

	__syncthreads();

	//Sort morton codes 
	u32* blockSharedMem = reinterpret_cast<u32*>(ptrAabbMem);
	for (uint32_t i = 0; i < 32; ++i)
	{
		u32 mortonCodeKey = mortonCodeKeys[primIdx];
		u32 mortonCodeValue = mortonCodeValues[primIdx];
		u32 bit = (mortonCodeKey >> i) & 1;
		u32 blockSum = binaryBlockPrefixSum(bit == 0, blockSharedMem);
		u32 newPrimIndex = bit == 0 ? blockSum - 1 : blockSharedMem[warpsPerBlock - 1] + primIdx - blockSum;
		__syncthreads();
		mortonCodeKeys[newPrimIndex] = mortonCodeKey;
		mortonCodeValues[newPrimIndex] = mortonCodeValue;
		__syncthreads();
	}

    //build bvh
	Bvh2Node* ptrBvhNodes = bvhNodes + bvhNodeOffset;
	PrimRef* ptrLeafNodes = bvhLeafNodes + bvhLeafNodeOffset;
	uint2* ptrSpans = spans + bvhNodeOffset;
	BvhBuildAndFit(primIdx, ptrBvhNodes, ptrLeafNodes, bvhNodeCounter, ptrSpans, mortonCodeKeys, primCount, primCount - 1);
	__syncthreads();

	rootNodes[gIdx] = bvhNodeCounter[primCount - 1];
}
