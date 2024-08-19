#pragma once
#include <src/Kernel.h>
#include <src/Context.h>
#include <src/Timer.h>
#include <src/Common.h>
#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <dependencies/Orochi/Orochi/GpuMemory.h>
#include <vector>

namespace BvhConstruction
{
	class HPLOC
	{
	public:
		void build(Context& context, std::vector<Triangle>& primitives);

		void traverseBvh(Context& context);

		Oro::GpuMemory<Triangle> d_triangleBuff;
		Oro::GpuMemory<Aabb> d_triangleAabb;
		Oro::GpuMemory<Aabb> d_sceneExtents;
		Oro::GpuMemory<u32> d_mortonCodeKeys;
		Oro::GpuMemory<u32> d_mortonCodeValues;
		Oro::GpuMemory<u32> d_sortedMortonCodeKeys;
		Oro::GpuMemory<u32> d_sortedMortonCodeValues;
		Oro::GpuMemory<Bvh2Node> d_bvhNodes;
		Oro::GpuMemory<PrimRef> d_leafNodes;

		u32 m_rootNodeIdx = 0;
		Timer m_timer;
		u32 m_nInternalNodes = 0;
		float m_cost = 0.0f;
	};
}