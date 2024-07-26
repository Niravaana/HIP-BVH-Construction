
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

	float3 centre = bounds[gIdx].center();
	float3 extents = sceneExtents[0].extent();
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

	if (gIdx >= width) return;
	if (gIdy >= height) return;

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
