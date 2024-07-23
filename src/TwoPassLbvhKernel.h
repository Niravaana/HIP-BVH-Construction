#include <src/Common.h>

using namespace BvhConstruction;

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

extern "C" __global__ void InitBvhNodes(
	const Triangle* __restrict__ primitives,
	LbvhNode32* __restrict__ bvhNodes,
	const u32* __restrict__ primIdx,
	const u32 nInternalNodes,
	const u32 nLeafNodes)
{
	unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (gIdx < nLeafNodes)
	{
		const u32 nodeIdx = gIdx + nInternalNodes;
		u32 idx = primIdx[gIdx];
		LbvhNode32& node = bvhNodes[nodeIdx];
		node.m_aabb.reset();
		node.m_aabb.grow(primitives[idx].v1); node.m_aabb.grow(primitives[idx].v2); node.m_aabb.grow(primitives[idx].v3);
		node.m_primIdx = idx;
		node.m_parentIdx = INVALID_NODE_IDX;
	}

	if (gIdx < nInternalNodes)
	{
		LbvhNode32& node = bvhNodes[gIdx];
		node.m_aabb.reset();
		node.m_leftChildIdx = INVALID_NODE_IDX;
		node.m_parentIdx = INVALID_NODE_IDX;
	}
}

extern "C" __global__ void BvhBuild(
	LbvhNode32* __restrict__ bvhNodes, 
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
	bvhNodes[leftChildIdx].m_parentIdx = gIdx;
	bvhNodes[rightChildIdx].m_parentIdx = gIdx;
}

extern "C" __global__ void FitBvhNodes(LbvhNode32* __restrict__ bvhNodes, u32* flags, u32 nLeafNodes,u32 nInternalNodes)
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
			u32 rightChildIdx = leftChildIdx + 1;
			bvhNodes[parent].m_aabb = merge(bvhNodes[leftChildIdx].m_aabb, bvhNodes[rightChildIdx].m_aabb);
			parent = bvhNodes[parent].m_parentIdx;
		}
		__threadfence();
	}
}

extern "C" __global__ void BvhTraversalifif(const  Ray* __restrict__ raysBuff, const  Triangle* __restrict__ primitives, const LbvhNode32* __restrict__ bvhNodes, const Transformation* __restrict__ tr, u8* __restrict__ colorBuffOut, u32 rootIdx, const u32 width, const u32 height, const u32 nInternalNodes)
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	if (gIdx >= width) return;
	if (gIdy >= height) return;

	u32 index = gIdx * width + gIdy;
	const Ray ray = raysBuff[index];
	u32 nodeIdx = rootIdx;
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
		const LbvhNode32& node = bvhNodes[nodeIdx];

		if (nodeIdx >= nInternalNodes)
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
			const Aabb right = bvhNodes[node.m_leftChildIdx + 1].m_aabb;
			const float2 t0 = left.intersect(transformedRay.m_origin, invRayDir, hit.m_t);
			const float2 t1 = right.intersect(transformedRay.m_origin, invRayDir, hit.m_t);

			const bool hitLeft = (t0.x <= t0.y);
			const bool hitRight = (t1.x <= t1.y);

			if (hitLeft || hitRight)
			{
				if (hitLeft && hitRight)
				{
					nodeIdx = (t0.x < t1.x) ? node.m_leftChildIdx : node.m_leftChildIdx + 1;
					if (top < 64)
					{
						stack[top++] = (t0.x < t1.x) ? node.m_leftChildIdx + 1 : node.m_leftChildIdx;
					}
				}
				else
				{
					nodeIdx = (hitLeft) ? node.m_leftChildIdx : node.m_leftChildIdx + 1;
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
