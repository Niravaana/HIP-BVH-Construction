
#include <src/Common.h>

using namespace BvhConstruction;

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

DEVICE uint64_t findHighestDiffBit(const u32* __restrict__ mortonCodes, int i, int j, int n)
{
	if (j < 0 || j >= n) return ~0ull;
	const uint64_t a = (static_cast<uint64_t>(mortonCodes[i]) << 32ull) | i;
	const uint64_t b = (static_cast<uint64_t>(mortonCodes[j]) << 32ull) | j;
	return a ^ b;
}

DEVICE int findParent(
	LbvhNode* __restrict__ bvhNodes,
	uint2* spans,
	const u32* __restrict__ mortonCodes,
	u32 currentNodeIdx,
	int i,
	int j,
	int n)
{
	if (i == 0 && j == n) return INVALID_NODE_IDX;
	if (i == 0 || (j != n && findHighestDiffBit(mortonCodes, j - 1, j, n) < findHighestDiffBit(mortonCodes, j - 1, j, n)))
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

extern "C" __global__ void BvhBuildAndFit(
	LbvhNode* __restrict__ bvhNodes,
	int* __restrict__ bvhNodeCounter,
	uint2* __restrict__ spans,
	const u32* __restrict__ mortonCodes,
	u32 nLeafNodes,
	u32 nInternalNodes)
{
	int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= nLeafNodes) return;

	int i = gIdx;
	int j = gIdx + 1;

	int parentIdx = findParent(bvhNodes, spans, mortonCodes, nInternalNodes + gIdx, i, j, nLeafNodes);
	gIdx = parentIdx;

	while (atomicAdd(&bvhNodeCounter[gIdx], 1) > 0)
	{
		__threadfence();

		LbvhNode& node = bvhNodes[gIdx];
		uint2 span = spans[gIdx];

		node.m_aabb = merge(bvhNodes[node.m_leftChildIdx].m_aabb, bvhNodes[node.m_rightChildIdx].m_aabb);

		parentIdx = findParent(bvhNodes, spans, mortonCodes, gIdx, span.x, span.y, nLeafNodes);

		if (parentIdx == INVALID_NODE_IDX)
		{
			bvhNodeCounter[nLeafNodes - 1] = gIdx; //Saving root node;
			break;
		}

		gIdx = parentIdx;

		__threadfence();
	}
}

extern "C" __global__ void BvhTraversalifif(const  Ray* __restrict__ raysBuff, const  Triangle* __restrict__ primitives, const LbvhNode* __restrict__ bvhNodes, const Transformation* __restrict__ tr, u8* __restrict__ colorBuffOut, u32 rootIdx, const u32 width, const u32 height)
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
	u32 stack[32];
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