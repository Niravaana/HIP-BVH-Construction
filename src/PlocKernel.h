
#include <src/Common.h>

using namespace BvhConstruction;

extern "C" __global__ void CalculateMortonCodes(const Aabb* __restrict__ primBounds, const Aabb* __restrict__ bounds, const Aabb* __restrict__ sceneExtents, u32* __restrict__ mortonCodesOut, u32* __restrict__ primIdxOut, u32 primCount)
{
	const unsigned int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gIdx >= primCount) return;

	float3 centre = primBounds[gIdx].center();
	float3 extents = sceneExtents[0].extent();
	float3 normalizedCentroid = (centre - sceneExtents[0].m_min) / extents;

	mortonCodesOut[gIdx] = computeExtendedMortonCode(normalizedCentroid, extents);
	primIdxOut[gIdx] = gIdx + (primCount - 1);
	bounds[gIdx + (primCount - 1)] = primBounds[gIdx];
}