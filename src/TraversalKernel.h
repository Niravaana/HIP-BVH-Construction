
#include <src/Common.h>

using namespace BvhConstruction;
#define __SHARED_STACK 1 

extern "C" __global__ void BvhTraversalifif(const  Ray* __restrict__ raysBuff, u32* rayCounter, const  Triangle* __restrict__ primitives, const LbvhNode* __restrict__ bvhNodes, const Transformation* __restrict__ tr, u8* __restrict__ colorBuffOut, u32 rootIdx, const u32 width, const u32 height, const u32 nInternalNodes)
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	if (gIdx >= width) return;
	if (gIdy >= height) return;

	u32 index = gIdx * width + gIdy;
	const Ray ray = raysBuff[index];
	u32 nodeIdx = rootIdx;
	u32 top = 0;
	constexpr int STACK_SIZE = 32;

#ifndef __SHARED_STACK
	u32 stack[STACK_SIZE];
#else
	constexpr int WG_SIZE = 64;
	int lIdx = blockDim.x * threadIdx.y + threadIdx.x;
	__shared__ u32 ldsBuffer[STACK_SIZE * WG_SIZE];
	u32* stack = &ldsBuffer[STACK_SIZE * lIdx];
#endif

	stack[top++] = INVALID_NODE_IDX;
	HitInfo hit;

	Ray transformedRay;
	transformedRay.m_origin = invTransform(ray.m_origin, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
	transformedRay.m_direction = invTransform(ray.m_direction, tr[0].m_scale, tr[0].m_quat, { 0.0f,0.0f,0.0f });
	float3 invRayDir = 1.0f / transformedRay.m_direction;

	while (nodeIdx != INVALID_NODE_IDX)
	{
		const LbvhNode& node = bvhNodes[nodeIdx];

		if (nodeIdx >= nInternalNodes)
		{
			const Triangle& triangle = primitives[node.m_leftChildIdx];
			float3 tV0 = transform(triangle.v1, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
			float3 tV1 = transform(triangle.v2, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
			float3 tV2 = transform(triangle.v3, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);

			float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);
			rayCounter[index]++;
			if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
			{
				hit.m_primIdx = node.m_leftChildIdx;
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

extern "C" __global__ void BvhTraversalWhile(const  Ray* __restrict__ raysBuff, const  Triangle* __restrict__ primitives, const LbvhNode* __restrict__ bvhNodes, const Transformation* __restrict__ tr, u8* __restrict__ colorBuffOut, u32 rootIdx, const u32 width, const u32 height, const u32 nInternalNodes)
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	if (gIdx >= width) return;
	if (gIdy >= height) return;

	u32 index = gIdx * width + gIdy;
	const Ray ray = raysBuff[index];
	u32 nodeIdx = rootIdx;
	u32 top = 0;
	constexpr int STACK_SIZE = 32;

#ifndef __SHARED_STACK
	u32 stack[STACK_SIZE];
#else
	constexpr int WG_SIZE = 64;
	int lIdx = blockDim.x * threadIdx.y + threadIdx.x;
	__shared__ u32 ldsBuffer[STACK_SIZE * WG_SIZE];
	u32* stack = &ldsBuffer[STACK_SIZE * lIdx];
#endif

	stack[top++] = INVALID_NODE_IDX;
	HitInfo hit;

	Ray transformedRay;
	transformedRay.m_origin = invTransform(ray.m_origin, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
	transformedRay.m_direction = invTransform(ray.m_direction, tr[0].m_scale, tr[0].m_quat, { 0.0f,0.0f,0.0f });
	float3 invRayDir = 1.0f / transformedRay.m_direction;

	while (nodeIdx != INVALID_NODE_IDX)
	{
		while (nodeIdx < nInternalNodes)
		{
			const LbvhNode& node = bvhNodes[nodeIdx];
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
			nodeIdx = stack[--top];
		}


		while (nodeIdx >= nInternalNodes && nodeIdx != INVALID_NODE_IDX)
		{
			const LbvhNode& node = bvhNodes[nodeIdx];

			if (nodeIdx >= nInternalNodes)
			{
				const Triangle& triangle = primitives[node.m_leftChildIdx];
				float3 tV0 = transform(triangle.v1, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
				float3 tV1 = transform(triangle.v2, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);
				float3 tV2 = transform(triangle.v3, tr[0].m_scale, tr[0].m_quat, tr[0].m_translation);

				float4 itr = intersectTriangle(tV0, tV1, tV2, ray.m_origin, ray.m_direction);

				if (itr.x > 0.0f && itr.y > 0.0f && itr.z > 0.0f && itr.w > 0.0f && itr.w < hit.m_t)
				{
					hit.m_primIdx = node.m_leftChildIdx;
					hit.m_t = itr.w;
					hit.m_uv = { itr.x, itr.y };
				}
			}

			nodeIdx = stack[--top];
		}
	}

	if (hit.m_primIdx != INVALID_PRIM_IDX)
	{
		colorBuffOut[index * 4 + 0] = (hit.m_uv.x) * 255;
		colorBuffOut[index * 4 + 1] = (hit.m_uv.y) * 255;
		colorBuffOut[index * 4 + 2] = (1 - hit.m_uv.x - hit.m_uv.y) * 255;
		colorBuffOut[index * 4 + 3] = 255;
	}
}

