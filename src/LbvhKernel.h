
#include <src/Common.h>

using namespace BvhConstruction;

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


extern "C" __global__ void CalculateSceneExtents(const Triangle* __restrict__ primitives, Aabb* __restrict__ primitivesAabb, Aabb* __restrict__ sceneExtent, u32 primCount)
{
    u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	sceneExtent->reset();
    Aabb primAabb;
    
	if (gIdx < primCount)
    {
        primAabb.grow(primitives[gIdx].v1); primAabb.grow(primitives[gIdx].v2); primAabb.grow(primitives[gIdx].v3);
		primitivesAabb[gIdx] = primAabb;
	}
    else
    {
        primAabb.grow(primitives[primCount - 1].v1); primAabb.grow(primitives[primCount - 1].v2); primAabb.grow(primitives[primCount - 1].v3);
    }

	__shared__ u8 sharedMem[sizeof(Aabb) * WarpSize];
	Aabb* pSharedMem = reinterpret_cast<Aabb*>(sharedMem);
	Aabb blockExtent = blockReduce(primAabb, pSharedMem);

    if (threadIdx.x == 0)
        sceneExtent->atomicGrow(blockExtent);
}

DEVICE u32 shiftBy2Space(u32 x)
{
	x = (x * 0x00010001u) & 0xFF0000FFu;
	x = (x * 0x00000101u) & 0x0F00F00Fu;
	x = (x * 0x00000011u) & 0xC30C30C3u;
	x = (x * 0x00000005u) & 0x49249249u;
	return x;
}

extern "C" __global__ void CalculateMortonCodes(const Aabb* __restrict__ bounds, const Aabb* __restrict__ sceneExtents, u32* __restrict__ mortonCodesOut, u32* __restrict__ primIdxOut, u32 primCount)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= primCount) return;

	Aabb aabb = bounds[gIdx];
	float3 centre = (aabb.m_min + aabb.m_max) * 0.5f;
	float3 extents = float3{ sceneExtents[0].m_max.x - sceneExtents[0].m_min.x, sceneExtents[0].m_max.y - sceneExtents[0].m_min.y, sceneExtents[0].m_max.z - sceneExtents[0].m_min.z };
	float3 normalizedCentroid = (centre - sceneExtents[0].m_min) / extents;

	float	 x = min(max(normalizedCentroid.x * 1024.0f, 0.0f), 1023.0f);
	float	 y = min(max(normalizedCentroid.y * 1024.0f, 0.0f), 1023.0f);
	float	 z = min(max(normalizedCentroid.z * 1024.0f, 0.0f), 1023.0f);

	u32 xx = shiftBy2Space(u32(x));
	u32 yy = shiftBy2Space(u32(y));
	u32 zz = shiftBy2Space(u32(z));

	mortonCodesOut[gIdx] = xx * 4 + yy * 2 + zz;
	primIdxOut[gIdx] = gIdx;
}

extern "C" __global__ void InitBvhNodes(
	LbvhInternalNode* __restrict__ internalNodes, 
	LbvhLeafNode* __restrict__  leafNodes,
	const u32* __restrict__ primIdx, 
	const u32 nInternalNodes, 
	const u32 nLeafNodes)
{
	unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (gIdx < nLeafNodes)
	{
		const u32 nodeIdx = gIdx;
		u32 idx = primIdx[nodeIdx];
		LbvhLeafNode& node = leafNodes[nodeIdx];
		node.m_primIdx = idx;
		node.m_shapeIdx = INVALID_NODE_IDX;
		node.m_parentIdx = INVALID_NODE_IDX;
	}

	if (gIdx < nInternalNodes) 
	{
		LbvhInternalNode& node = internalNodes[gIdx];
		node.m_rAabb.reset();
		node.m_lAabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_rightChildIdx = INVALID_NODE_IDX;
		node.m_parentIdx = INVALID_NODE_IDX;
	}
}