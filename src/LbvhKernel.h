
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
	if (firstCode == lastCode)
	{
		return (first + last) >> 1;
	}
	const u32 deltaNode = countCommonPrefixBits(firstCode, lastCode);

	// binary search
	int split = first;
	int stride = last - first;
	do
	{
		stride = (stride + 1) >> 1;
		const int middle = split + stride;
		if (middle < last)
		{
			const u32 delta = countCommonPrefixBits(firstCode, mortonCode[middle]);
			if (delta > deltaNode)
			{
				split = middle;
			}
		}
	} while (stride > 1);

	return split;
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

extern "C" __global__ void FitBvhNodes(LbvhNode* __restrict__ bvhNodes, u32* flags, u32 nLeafNodes,u32 nInternalNodes)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= nLeafNodes) return;

	int idx = nInternalNodes + gIdx;
	u32 parent = bvhNodes[idx].m_parentIdx;

	while (parent != INVALID_NODE_IDX)
	{
		if (atomicCAS(&flags[parent], 0, 1) == 0)
			break;

		__threadfence();
		{
			u32 leftChildIdx = bvhNodes[parent].m_leftChildIdx;
			u32 rightChildIdx = bvhNodes[parent].m_rightChildIdx;
			bvhNodes[parent].m_aabb = merge(bvhNodes[leftChildIdx].m_aabb, bvhNodes[rightChildIdx].m_aabb);
			parent = bvhNodes[parent].m_parentIdx;
		}
		__threadfence();
	}
}

HOST_DEVICE INLINE uint32_t lcg(uint32_t& seed)
{
	const uint32_t LCG_A = 1103515245u;
	const uint32_t LCG_C = 12345u;
	const uint32_t LCG_M = 0x00FFFFFFu;
	seed = (LCG_A * seed + LCG_C);
	return seed & LCG_M;
}

HOST_DEVICE INLINE float randf(uint32_t& seed)
{
	return (static_cast<float>(lcg(seed)) / static_cast<float>(0x01000000));
}

template <uint32_t N>
HOST_DEVICE INLINE uint2 tea(uint32_t val0, uint32_t val1)
{
	uint32_t v0 = val0;
	uint32_t v1 = val1;
	uint32_t s0 = 0;

	for (uint32_t n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return { v0, v1 };
}

extern "C" __global__ void GenerateRays(const Camera* __restrict__ cam, Ray* __restrict__ raysBuffOut, const u32 width, const u32 height)
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	//if (gIdx >= width) return;
	//if (gIdy >= height) return;

	bool isMultiSamples = false;
	float  fov = cam->m_fov;
	float2 sensorSize;
	unsigned int seed = tea<16>(gIdx + gIdy * width, 0).x;

	sensorSize.x = 0.024f * (width / static_cast<float>(height));
	sensorSize.y = 0.024f;
	float		 offset = (isMultiSamples) ? randf(seed) : 0.5f;
	const float2 xy = float2{ ((float)gIdx + offset) / width, ((float)gIdy + offset) / height } - float2{ 0.5f, 0.5f };
	float3		 dir = float3{ xy.x * sensorSize.x, xy.y * sensorSize.y, sensorSize.y / (2.f * tan(fov / 2.f)) };

	const float3 holDir = qtRotate(cam->m_quat, float3{ 1.0f, 0.0f, 0.0f });
	const float3 upDir = qtRotate(cam->m_quat, float3{ 0.0f, -1.0f, 0.0f });
	const float3 viewDir = qtRotate(cam->m_quat, float3{ 0.0f, 0.0f, -1.0f });
	dir = normalize(dir.x * holDir + dir.y * upDir + dir.z * viewDir);

	{
		raysBuffOut[gIdx * height + gIdy].m_origin = float3{ cam->m_eye.x, cam->m_eye.y, cam->m_eye.z };
		float4 direction = cam->m_eye + float4{ dir.x * cam->m_far, dir.y * cam->m_far, dir.z * cam->m_far, 0.0f };
		raysBuffOut[gIdx * height + gIdy].m_direction = normalize(float3{ direction.x, direction.y, direction.z });
		raysBuffOut[gIdx * height + gIdy].m_tMin = 0.0f;
		raysBuffOut[gIdx * height + gIdy].m_tMax = FltMax;
	}
}

extern "C" __global__ void BvhTraversal(const  Ray* raysBuff, const  Triangle* primitives, const LbvhNode* bvhNodes, Transformation* tr,  u8* colorBuffOut, const u32 width, const u32 height)
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	if (gIdx >= width) return;
	if (gIdy >= height) return;

	u32 index = gIdx * width + gIdy;
	const Ray ray = raysBuff[index];
	u32 nodeIdx = 0;
	u32 top = 0;
	u32 stack[64];
	stack[top++] = INVALID_NODE_IDX;
	HitInfo hit;

	Ray transformedRay;
	transformedRay.m_origin = invTransform(ray.m_origin, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
	transformedRay.m_direction = invTransform(ray.m_direction, tr[0].m_scale, tr[0].m_quat, { 0.0f,0.0f,0.0f });
	float3 invRayDir = 1.0f / transformedRay.m_direction;

	while (nodeIdx != INVALID_NODE_IDX)
	{
		const LbvhNode& node = bvhNodes[nodeIdx];

		if (LbvhNode::isLeafNode(node))
		{
			const Triangle& triangle = primitives[node.m_primIdx];
			float3 tV0 = transform(triangle.v1, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
			float3 tV1 = transform(triangle.v2, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
			float3 tV2 = transform(triangle.v3, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);

			float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);

			if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
			{
				hit.m_primIdx = node.m_primIdx;
				hit.m_t = itr.w;
				hit.m_uv = { itr.x, itr.y };
			}
		}
		else
		{
			const Aabb left = bvhNodes[node.m_leftChildIdx].m_aabb;
			const Aabb right = bvhNodes[node.m_rightChildIdx].m_aabb;
			const float2 t0 = left.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
			const float2 t1 = right.intersect(transformedRay.m_origin, invRayDir, hit.m_t);

			const bool hitLeft = (t0.x <= t0.y);
			const bool hitRight = (t1.x <= t1.y);

			if (hitLeft || hitRight)
			{
				if (hitLeft && hitRight)
				{
					nodeIdx = (t0.x < t1.x) ? node.m_leftChildIdx : node.m_rightChildIdx;
					if (top < 64)
					{
						stack[top++] = (t0.x < t1.x) ? node.m_rightChildIdx : node.m_leftChildIdx;
					}
				}
				else
				{
					nodeIdx = (hitLeft) ? node.m_leftChildIdx : node.m_rightChildIdx;
				}
				continue;
			}
		}
		nodeIdx = stack[--top];
	}
	
	if (hit.m_primIdx != INVALID_PRIM_IDX)
	{
		colorBuffOut[index * 4 + 0] = (hit.m_uv.x) * 255;
		colorBuffOut[index * 4 + 1] = (hit.m_uv.y) * 255;
		colorBuffOut[index * 4 + 2] = (1 - hit.m_uv.x - hit.m_uv.y) * 255;
		colorBuffOut[index * 4 + 3] = 255;
	}
}
