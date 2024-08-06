
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


extern "C" __global__ void InitPrimRefs(PrimRef* __restrict__ primRefs, const Triangle* __restrict__ primitives, u32 primCount)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= primCount) return;

	Aabb primAabb;
	primAabb.grow(primitives[gIdx].v1); primAabb.grow(primitives[gIdx].v2); primAabb.grow(primitives[gIdx].v3);
	primRefs[gIdx].m_primIdx = gIdx;
	primRefs[gIdx].m_aabb = primAabb;
}

extern "C" __global__ void CalculatePrimRefExtents(PrimRef* __restrict__ primRefs, Aabb* __restrict__ sceneExtent, u32 primCount)
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	sceneExtent->reset();
	Aabb primAabb;

	if (gIdx < primCount)
	{
		primAabb = primRefs[gIdx].m_aabb;
	}
	else
	{
		primAabb = primRefs[primCount - 1].m_aabb;
	}

	__shared__ u8 sharedMem[sizeof(Aabb) * WarpSize];
	Aabb* pSharedMem = reinterpret_cast<Aabb*>(sharedMem);
	Aabb blockExtent = blockReduce(primAabb, pSharedMem);

	if (threadIdx.x == 0)
		sceneExtent->atomicGrow(blockExtent);
}

DEVICE u32 morton2D(u32 v)
{
	v &= 0x0000ffff;				
	v = (v ^ (v << 8)) & 0x00ff00ff;
	v = (v ^ (v << 4)) & 0x0f0f0f0f;
	v = (v ^ (v << 2)) & 0x33333333;
	v = (v ^ (v << 1)) & 0x55555555;
	return v;
}

DEVICE u32 morton3D(u32 x)
{
	x = (x * 0x00010001u) & 0xFF0000FFu;
	x = (x * 0x00000101u) & 0x0F00F00Fu;
	x = (x * 0x00000011u) & 0xC30C30C3u;
	x = (x * 0x00000005u) & 0x49249249u;
	return x;
}


DEVICE u32 computeExtendedMortonCode(float3 normalizedPos, float3 sceneExtent)
{
	const u32 numMortonBits = 30;
	int3		   numBits = { 0, 0, 0 };

	int3 numPrebits;
	int3 startAxis;

	if (sceneExtent.x < sceneExtent.y)
	{
		if (sceneExtent.x < sceneExtent.z)
		{
			if (sceneExtent.y < sceneExtent.z)
			{
				// z, y, x
				startAxis.x = 2;
				numPrebits.x = log2(sceneExtent.z / sceneExtent.y);

				startAxis.y = 1;
				numPrebits.y = log2(sceneExtent.y / sceneExtent.x);

				startAxis.z = 0;
				numPrebits.z = log2(sceneExtent.z / sceneExtent.x);
			}
			else
			{
				// y, z, x
				startAxis.x = 1;
				numPrebits.x = log2(sceneExtent.y / sceneExtent.z);

				startAxis.y = 2;
				numPrebits.y = log2(sceneExtent.z / sceneExtent.x);

				startAxis.z = 0;
				numPrebits.z = log2(sceneExtent.y / sceneExtent.x);
			}
		}
		else
		{
			// y, x, z
			startAxis.x = 1;
			numPrebits.x = log2(sceneExtent.y / sceneExtent.x);

			startAxis.y = 0;
			numPrebits.y = log2(sceneExtent.x / sceneExtent.z);

			startAxis.z = 2;
			numPrebits.z = log2(sceneExtent.y / sceneExtent.z);
		}
	}
	else
	{
		if (sceneExtent.y < sceneExtent.z)
		{
			if (sceneExtent.x < sceneExtent.z)
			{
				// z, x, y
				startAxis.x = 2;
				numPrebits.x = log2(sceneExtent.z / sceneExtent.x);

				startAxis.y = 0;
				numPrebits.y = log2(sceneExtent.x / sceneExtent.y);

				startAxis.z = 1;
				numPrebits.z = log2(sceneExtent.z / sceneExtent.y);
			}
			else
			{
				// x, z, y
				startAxis.x = 0;
				numPrebits.x = log2(sceneExtent.x / sceneExtent.z);

				startAxis.y = 2;
				numPrebits.y = log2(sceneExtent.z / sceneExtent.y);

				startAxis.z = 1;
				numPrebits.z = log2(sceneExtent.x / sceneExtent.y);
			}
		}
		else
		{
			// x, y, z
			startAxis.x = 0;
			numPrebits.x = log2(sceneExtent.x / sceneExtent.y);

			startAxis.y = 1;
			numPrebits.y = log2(sceneExtent.y / sceneExtent.z);

			startAxis.z = 2;
			numPrebits.z = log2(sceneExtent.x / sceneExtent.z);
		}
	}

	int swap = numPrebits.z - (numPrebits.x + numPrebits.y);

	numPrebits.x = min(numPrebits.x, numMortonBits);
	numPrebits.y = min(numPrebits.y * 2, numMortonBits - numPrebits.x) / 2;

	int numPrebitsSum = numPrebits.x + numPrebits.y * 2;

	if (numPrebitsSum != numMortonBits)
		numPrebitsSum += swap;
	else
		swap = 0;

	numBits.z = ((&sceneExtent.x)[startAxis.z] != 0) ? max(0, (numMortonBits - numPrebitsSum) / 3) : 0;

	if (swap > 0)
	{
		numBits.x = max(0, (numMortonBits - numBits.z - numPrebitsSum) / 2 + numPrebits.y + numPrebits.x + 1);
		numBits.y = numMortonBits - numBits.x - numBits.z;
	}
	else
	{
		numBits.y = max(0, (numMortonBits - numBits.z - numPrebitsSum) / 2 + numPrebits.y);
		numBits.x = numMortonBits - numBits.y - numBits.z;
	}

	u32 mortonCode = 0;
	int3	 axisCode;

	// Based on the number of bits, calculate each code per axis
	axisCode.x = min(u32(max((&normalizedPos.x)[startAxis.x] * (1u << numBits.x), 0.0f)), (1u << numBits.x) - 1);
	axisCode.y = min(u32(max((&normalizedPos.x)[startAxis.y] * (1u << numBits.y), 0.0f)), (1u << numBits.y) - 1);
	axisCode.z = min(u32(max((&normalizedPos.x)[startAxis.z] * (1u << numBits.z), 0.0f)), (1u << numBits.z) - 1);

	u32 delta0 = 0;
	u32 delta1 = 0;


	if (numPrebitsSum > 0)
	{
		numBits.x -= numPrebits.x;
		mortonCode = axisCode.x & (((1U << numPrebits.x) - 1) << numBits.x);
		mortonCode >>= numBits.x;

		mortonCode <<= numPrebits.y * 2;
		numBits.x -= numPrebits.y;
		numBits.y -= numPrebits.y;
		u32 temp0 = axisCode.x & (((1u << numPrebits.y) - 1) << numBits.x);
		temp0 >>= numBits.x;
		temp0 = morton2D(temp0);

		u32 temp1 = axisCode.y & (((1u << numPrebits.y) - 1) << numBits.y);
		temp1 >>= numBits.y;
		temp1 = morton2D(temp1);

		mortonCode |= temp0 * 2 + temp1;

		if (swap > 0)
		{
			mortonCode <<= 1;
			numBits.x -= 1;
			u32 temp = axisCode.x & (1U << numBits.x);
			temp >>= numBits.x;
			mortonCode |= temp;
		}

		mortonCode <<= numBits.x + numBits.y + numBits.z;

		axisCode.x &= ((1u << numBits.x) - 1);
		axisCode.y &= ((1u << numBits.y) - 1);

		if (swap > 0)
		{
			delta0 = (numBits.y - numBits.x);
			axisCode.x <<= delta0;

			delta1 = (numBits.y - numBits.z);
			axisCode.z <<= delta1;
		}
		else
		{
			delta0 = (numBits.x - numBits.y);
			axisCode.y <<= delta0;

			delta1 = (numBits.x - numBits.z);
			axisCode.z <<= delta1;
		}
	}

	if (numBits.z == 0)
	{
		axisCode.x = morton2D(axisCode.x);
		axisCode.y = morton2D(axisCode.y);
		mortonCode |= axisCode.x * 2 + axisCode.y;
	}
	else
	{
		axisCode.x = (axisCode.x > 0) ? morton3D(axisCode.x) : 0;
		axisCode.y = (axisCode.y > 0) ? morton3D(axisCode.y) : 0;
		axisCode.z = (axisCode.z > 0) ? morton3D(axisCode.z) : 0;

		if (swap > 0)
			mortonCode |= (axisCode.y * 4 + axisCode.x * 2 + axisCode.z) >> (delta0 + delta1);
		else
			mortonCode |= (axisCode.x * 4 + axisCode.y * 2 + axisCode.z) >> (delta0 + delta1);
	}

	return mortonCode;
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

extern "C" __global__ void CalculateMortonCodes(const Aabb* __restrict__ bounds, const Aabb* __restrict__ sceneExtents, u32* __restrict__ mortonCodesOut, u32* __restrict__ primIdxOut, u32 primCount)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= primCount) return;

	float3 centre = bounds[gIdx].center();
	float3 extents = sceneExtents[0].extent();
	float3 normalizedCentroid = (centre - sceneExtents[0].m_min) / extents;

	mortonCodesOut[gIdx] = computeExtendedMortonCode(normalizedCentroid, extents);
	primIdxOut[gIdx] = gIdx;
}

extern "C" __global__ void CalculateMortonCodesPrimRef(const PrimRef* __restrict__ primRefs, const Aabb* __restrict__ sceneExtents, u32* __restrict__ mortonCodesOut, u32* __restrict__ primIdxOut, u32 primCount)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= primCount) return;

	float3 centre = primRefs[gIdx].m_aabb.center();
	float3 extents = sceneExtents[0].extent();
	float3 normalizedCentroid = (centre - sceneExtents[0].m_min) / extents;

	mortonCodesOut[gIdx] = computeExtendedMortonCode(normalizedCentroid, extents);
	primIdxOut[gIdx] = gIdx;
}


HOST_DEVICE INLINE u32 lcg(u32& seed)
{
	const u32 LCG_A = 1103515245u;
	const u32 LCG_C = 12345u;
	const u32 LCG_M = 0x00FFFFFFu;
	seed = (LCG_A * seed + LCG_C);
	return seed & LCG_M;
}

HOST_DEVICE INLINE float randf(u32& seed)
{
	return (static_cast<float>(lcg(seed)) / static_cast<float>(0x01000000));
}

template <u32 N>
HOST_DEVICE INLINE uint2 tea(u32 val0, u32 val1)
{
	u32 v0 = val0;
	u32 v1 = val1;
	u32 s0 = 0;

	for (u32 n = 0; n < N; n++)
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
