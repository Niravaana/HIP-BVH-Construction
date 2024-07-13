#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#if ( defined( __CUDACC_RTC__ ) || defined( __HIPCC_RTC__ ) )
#define __KERNELCC_RTC__
#endif

#ifdef __KERNELCC__
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define INLINE __forceinline__
#else
#define HOST 
#define DEVICE 
#define HOST_DEVICE 
#define INLINE inline
#endif

#include <math.h>

namespace BvhConstruction
{
	constexpr float FltMin = 1.175494351e-38f;
	constexpr float FltMax = 3.402823466e+38f;
	constexpr int	IntMin = -2147483647 - 1;
	constexpr int	IntMax = 2147483647;

	enum Error
	{
		Success,
		Failure
	};

	using u32 = uint32_t;
	using u8 = unsigned char;
	using u64 = unsigned long long;

#ifdef __KERNELCC__
#if __gfx900__ || __gfx902__ || __gfx904__ || __gfx906__ || __gfx908__ || __gfx909__ || __gfx90a__ || __gfx90c__
	constexpr int WarpSize = 64;
#else
	constexpr int WarpSize = 32;
#endif
#endif

#ifndef __KERNELCC__
	struct float3
	{
		float x, y, z;
	};

	struct float4
	{
		float x, y, z, w;
	};

	struct float2
	{
		float x, y;
	};
#endif

	HOST_DEVICE INLINE float3 max(const float3& a, const float3& b)
	{
		float x = fmaxf(a.x, b.x);
		float y = fmaxf(a.y, b.y);
		float z = fmaxf(a.z, b.z);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 max(const float3& a, const float c)
	{
		float x = fmaxf(a.x, c);
		float y = fmaxf(a.y, c);
		float z = fmaxf(a.z, c);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 max(const float c, const float3& a)
	{
		float x = fmaxf(a.x, c);
		float y = fmaxf(a.y, c);
		float z = fmaxf(a.z, c);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 min(const float3& a, const float3& b)
	{
		float x = fminf(a.x, b.x);
		float y = fminf(a.y, b.y);
		float z = fminf(a.z, b.z);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 min(const float3& a, const float c)
	{
		float x = fminf(a.x, c);
		float y = fminf(a.y, c);
		float z = fminf(a.z, c);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 min(const float c, const float3& a)
	{
		float x = fminf(a.x, c);
		float y = fminf(a.y, c);
		float z = fminf(a.z, c);
		return float3{ x, y, z };
	}

	HOST_DEVICE INLINE float3 fma(const float3& a, const float3& b, const float3& c)
	{
		float x = fmaf(a.x, b.x, c.x);
		float y = fmaf(a.y, b.y, c.y);
		float z = fmaf(a.z, b.z, c.z);
		return float3{ (x, y, z) };
	}

	HOST_DEVICE INLINE float3 operator+(const float3& a, const float3& b)
	{
		return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
	}

	HOST_DEVICE INLINE float3 operator/(const float3& a, const float3& b)
	{
		return float3{ a.x / b.x, a.y / b.y, a.z / b.z };
	}

	HOST_DEVICE INLINE float3& operator*=(float3& a, const float c)
	{
		a.x *= c;
		a.y *= c;
		a.z *= c;
		return a;
	}

	HOST_DEVICE INLINE float3 operator*(const float c, const float3& a)
	{
		return float3{ c * a.x, c * a.y, c * a.z };
	}

	HOST_DEVICE INLINE float3 operator*(const float3& a, const float c)
	{
		return float3{ c * a.x, c * a.y, c * a.z };
	}

	HOST_DEVICE INLINE float3 operator-(const float3& a, const float3& b)
	{
		return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
	}

#if defined( __KERNELCC__ )
DEVICE INLINE float atomicMinFloat(float* addr, float value)
{
	float old;
	old = (__float_as_int(value) >= 0)
		? __int_as_float(atomicMin(reinterpret_cast<int*>(addr), __float_as_int(value)))
		: __uint_as_float(atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value)));
	return old;
}

DEVICE INLINE float atomicMaxFloat(float* addr, float value)
{
	float old;
	old = (__float_as_int(value) >= 0)
		? __int_as_float(atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value)))
		: __uint_as_float(atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value)));
	return old;
}
#endif

	class Aabb
	{
	public:
		HOST_DEVICE Aabb() { reset(); }

		DEVICE Aabb(const float3& p) : m_min(p), m_max(p) {}

		HOST_DEVICE Aabb(const float3& mi, const float3& ma) : m_min(mi), m_max(ma) {}

		HOST_DEVICE Aabb(const Aabb& rhs, const Aabb& lhs)
		{
			m_min = min(lhs.m_min, rhs.m_min);
			m_max = max(lhs.m_max, rhs.m_max);
		}

		HOST_DEVICE Aabb(const Aabb& rhs) : m_min(rhs.m_min), m_max(rhs.m_max) {}

		HOST_DEVICE void reset(void)
		{
			m_min = float3{ FltMax, FltMax, FltMax };
			m_max = float3{ -FltMax, -FltMax, -FltMax };
		}

		HOST_DEVICE Aabb& grow(const Aabb& rhs)
		{
			m_min = min(m_min, rhs.m_min);
			m_max = max(m_max, rhs.m_max);
			return *this;
		}

		HOST_DEVICE Aabb& grow(const float3& p)
		{
			m_min = min(m_min, p);
			m_max = max(m_max, p);
			return *this;
		}

		HOST_DEVICE float3 center() const { return (m_max + m_min) * 0.5f; }

		HOST_DEVICE float3 extent() const { return m_max - m_min; }

		HOST_DEVICE float area() const
		{
			float3 ext = extent();
			return 2 * (ext.x * ext.y + ext.x * ext.z + ext.y * ext.z);
		}

		HOST_DEVICE bool valid(void) { return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z; }

		HOST_DEVICE void intersect(const Aabb& box)
		{
			m_min = max(m_min, box.m_min);
			m_max = min(m_max, box.m_max);
		}
		
#if defined( __KERNELCC__ )
		DEVICE void atomicGrow(const Aabb& aabb)
		{
			atomicMinFloat(&m_min.x, aabb.m_min.x);
			atomicMinFloat(&m_min.y, aabb.m_min.y);
			atomicMinFloat(&m_min.z, aabb.m_min.z);
			atomicMaxFloat(&m_max.x, aabb.m_max.x);
			atomicMaxFloat(&m_max.y, aabb.m_max.y);
			atomicMaxFloat(&m_max.z, aabb.m_max.z);
		}
#endif

	public:
		float3 m_min;
		float3 m_max;
	};

	struct Triangle
	{
		float3 v1;
		float3 v2;
		float3 v3;
	};

	struct LbvhNode
	{
		u32 m_parentIdx;
		u32 m_leftChildIdx;
		u32 m_rightChildIdx;
		Aabb m_rAabb;
		Aabb m_lAabb;
		u32 m_primIdx;
	};

	constexpr size_t size = sizeof(LbvhNode);
	constexpr u32 INVALID_NODE_IDX = 0xFFFFFFFF;
	constexpr u32 INVALID_PRIM_IDX = 0xFFFFFFFF;

}