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
	using u32 = uint32_t;
	using u8 = unsigned char;
	using u64 = unsigned long long;

	constexpr float FltMin = 1.175494351e-38f;
	constexpr float FltMax = 3.402823466e+38f;
	constexpr int	IntMin = -2147483647 - 1;
	constexpr int	IntMax = 2147483647;
	constexpr float Pi = 3.14159265358979323846f;
	constexpr u32 INVALID_NODE_IDX = 0xFFFFFFFF;
	constexpr u32 INVALID_PRIM_IDX = 0xFFFFFFFF;

	enum Error
	{
		Success,
		Failure
	};

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

	struct uint2
	{
		u32 x, y;
	};

	HOST_DEVICE INLINE float3 operator+(const float3& a, const float3& b)
	{
		return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
	}

	HOST_DEVICE INLINE float4 operator+(const float4& a, const float4& b)
	{
		return float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
	}

	HOST_DEVICE INLINE float3 operator-(const float3& a, const float3& b)
	{
		return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
	}

	HOST_DEVICE INLINE float4 operator-(const float4& a, const float4& b)
	{
		return float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
	}

	HOST_DEVICE INLINE float2 operator-(const float2& a, const float2& b)
	{
		return float2{ a.x - b.x, a.y - b.y };
	}

	HOST_DEVICE INLINE float3 operator-(const float3& a)
	{
		return float3{ -a.x, -a.y, -a.z };
	}

	HOST_DEVICE INLINE float4 operator-(const float4& a)
	{
		return float4{ -a.x, -a.y, -a.z, -a.w };
	}

	HOST_DEVICE INLINE float3 operator/(const float3& a, const float3& b)
	{
		return float3{ a.x / b.x, a.y / b.y, a.z / b.z };
	}

	HOST_DEVICE INLINE float3 operator/(const float3& a, const float& b)
	{
		return float3{ a.x / b, a.y / b, a.z / b };
	}

	HOST_DEVICE INLINE float3 operator/(const float b, const float3& a)
	{
		return float3{ b / a.x , b / a.y , b / a.z };
	}

	HOST_DEVICE INLINE float4 operator/(const float4& a, const float& b)
	{
		return float4{ a.x / b, a.y / b, a.z / b, a.w / b };
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

	HOST_DEVICE INLINE float3 operator*(const float3& a, const float3& b)
	{
		return float3{ a.x * b.x, a.y * b.y, a.z * b.z };
	}

	HOST_DEVICE INLINE float3 operator*(const float3& a, const float c)
	{
		return float3{ c * a.x, c * a.y, c * a.z };
	}

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



	HOST_DEVICE INLINE float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	HOST_DEVICE INLINE float3 normalize(const float3& a) { return a / sqrtf(dot(a, a)); }

	HOST_DEVICE INLINE float3 cross(const float3& a, const float3& b)
	{
		return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
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

		HOST_DEVICE int maximumExtentDim() const {
			float3 d = extent();
			if (d.x > d.y && d.x > d.z)
				return 0;
			else if (d.y > d.z)
				return 1;
			else
				return 2;
		}

		HOST_DEVICE float area() const
		{
			float3 ext = extent();
			return 2 * (ext.x * ext.y + ext.x * ext.z + ext.y * ext.z);
		}

		HOST_DEVICE float3 offset(const float3& p) const
		{
			float3 o = p - m_min;
			if (m_max.x > m_min.x) o.x /= m_max.x - m_min.x;
			if (m_max.y > m_min.y) o.y /= m_max.y - m_min.y;
			if (m_max.z > m_min.z) o.z /= m_max.z - m_min.z;
			return o;
		}

		HOST_DEVICE bool valid(void) { return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z; }

		HOST_DEVICE void intersect(const Aabb& box)
		{
			m_min = max(m_min, box.m_min);
			m_max = min(m_max, box.m_max);
		}
		
		HOST_DEVICE float2 intersect(const float3& from, const float3& invRay, float maxt) const
		{
			const float3 dFar = (m_max - from) * (invRay);
			const float3 dNear = (m_min - from) * (invRay);
			const float3 tFar = max(dFar, dNear);
			const float3 tNear = min(dFar, dNear);
			float minFar = fmin(tFar.x, fmin(tFar.y, tFar.z));
			float maxNear = fmax(tNear.x, fmax(tNear.y, tNear.z));

			minFar = fmin(maxt, minFar);
			maxNear = fmax(0.0f, maxNear);

			return { maxNear, minFar };
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

	struct alignas(64) Triangle
	{
		float3 v1;
		float3 v2;
		float3 v3;
	};

	struct alignas(64) LbvhNode
	{
		u32 m_parentIdx;
		u32 m_leftChildIdx;
		u32 m_rightChildIdx;
		u32 m_primIdx;
		Aabb m_aabb;
		float m_pad1;
		float m_pad2;

		static HOST_DEVICE bool isLeafNode(const LbvhNode& node)
		{
			return (node.m_leftChildIdx == INVALID_NODE_IDX && node.m_rightChildIdx == INVALID_NODE_IDX && node.m_primIdx != INVALID_NODE_IDX);
		}
	};

	HOST_DEVICE INLINE Aabb merge(const Aabb lhs, const Aabb rhs)
	{
		return  { min(lhs.m_min, rhs.m_min), max(lhs.m_max, rhs.m_max) };
	}
	
	HOST_DEVICE INLINE float4 qtRotation(float4 axisAngle)
	{
		float3 axis = normalize(float3{ axisAngle.x, axisAngle.y, axisAngle.z });
		float  angle = axisAngle.w;

		float4 q;
		q.x = axis.x * sinf(angle / 2.0f);
		q.y = axis.y * sinf(angle / 2.0f);
		q.z = axis.z * sinf(angle / 2.0f);
		q.w = cosf(angle / 2.0f);
		return q;
	}

	HOST_DEVICE INLINE float4 qtGetIdentity(void) { return float4{ 0.0f, 0.0f, 0.0f, 1.0f }; }

	HOST_DEVICE INLINE float qtDot(const float4& q0, const float4& q1)
	{
		return q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w;
	}

	HOST_DEVICE INLINE float4 qtNormalize(const float4& q) { return q / sqrtf(qtDot(q, q)); }

	HOST_DEVICE INLINE float4 qtMul(const float4& a, const float4& b)
	{
		float4 ans;
		float3 aXb = cross(float3{ a.x, a.y, a.z }, float3{ b.x, b.y, b.z });
		ans = float4{ aXb.x, aXb.y, aXb.z, 0.0f };
		// ans += a.w * b + b.w * a;
		ans = ans + float4{ a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w } + float4{ b.w * a.x, b.w * a.y, b.w * a.z, b.w * a.w };
		ans.w = a.w * b.w - dot(float3{ a.x, a.y, a.z }, float3{ b.x, b.y, b.z });
		return ans;
	}

	HOST_DEVICE INLINE float4 qtInvert(const float4& q)
	{
		float4 ans;
		ans = -q;
		ans.w = q.w;
		return ans;
	}

	HOST_DEVICE INLINE float3 qtRotate(const float4& q, const float3& p)
	{
		float4 qp = float4{ p.x, p.y, p.z, 0.0f };
		float4 qInv = qtInvert(q);
		float4 out = qtMul(qtMul(q, qp), qInv);
		return float3{ out.x, out.y, out.z };
	}

	HOST_DEVICE INLINE  float3 qtInvRotate(const float4& q, const float3& vec) { return qtRotate(qtInvert(q), vec); }

	HOST_DEVICE INLINE  float3 invTransform(const float3& p, const float3& scale, const float4& rotation, const float3& translation) { return qtInvRotate(rotation, p - translation) / scale; }

	HOST_DEVICE INLINE  float3 transform(const float3& p, const float3& scale, const float4& rotation, const float3& translation) { return qtRotate(rotation, scale * p) + translation; }

	HOST_DEVICE INLINE  float4 intersectTriangle(const float3& v0, const float3& v1, const float3& v2, const float3& rayOrg, const float3& rayDir)
	{
		const float3 pos0 = v0 - rayOrg;
		const float3 pos1 = v1 - rayOrg;
		const float3 pos2 = v2 - rayOrg;
		const float3 edge0 = v2 - v0;
		const float3 edge1 = v0 - v1;
		const float3 edge2 = v1 - v2;
		const float3 normal = cross(edge1, edge0);
		const float u = dot(cross(pos0 + pos2, edge0), rayDir);
		const float v = dot(cross(pos1 + pos0, edge1), rayDir);
		const float w = dot(cross(pos2 + pos1, edge2), rayDir);
		const float t = dot(pos0, normal) * 2.0f;

		return float4{ u, v, w, t } / (dot(normal, rayDir) * 2.0f);
	}

	struct alignas(64) Ray
	{
		float3 m_origin;
		float3 m_direction;
		float m_tMin = 0.0f;
		float m_tMax = FltMax;
	};

	struct alignas(64) Transformation
	{
		float3 m_translation;
		float m_pad;
		float3 m_scale;
		float m_pad1;
		float4 m_quat;
	};

	struct alignas(64) Camera
	{
		float4	   m_eye; 
		float4	   m_quat;
		float	   m_fov;
		float	   m_near;
		float	   m_far;
		float	   m_pad;
	};

	struct alignas(64) HitInfo
	{
		u32 m_primIdx = INVALID_PRIM_IDX;
		float m_t = FltMax;
		float2 m_uv; //barycentric coordinates
	};

	constexpr size_t size = sizeof(Ray);

}