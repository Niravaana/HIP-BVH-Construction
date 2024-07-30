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
	class TwoPassLbvh
	{
	public:
		void build(Context& context, std::vector<Triangle>& primitives);

		void traverseBvh(Context& context);

		void collapseBvh2toBvh4(const std::vector< LbvhNode>& bvh2Nodes, std::vector<Bvh4Node>& bvh4Nodes, std::vector<PrimNode> bvh4LeafNodes, std::vector<uint2>& taskQ, u32 taskCount, u32 bvh8InternalNodeOffset, u32 nBvh2InternalNodes, u32 nBvh2LeafNodes);

		Oro::GpuMemory<Triangle> d_triangleBuff;
		Oro::GpuMemory<Aabb> d_triangleAabb;
		Oro::GpuMemory<Aabb> d_sceneExtents;
		Oro::GpuMemory<u32> d_mortonCodeKeys;
		Oro::GpuMemory<u32> d_mortonCodeValues;
		Oro::GpuMemory<u32> d_sortedMortonCodeKeys;
		Oro::GpuMemory<u32> d_sortedMortonCodeValues;
		Oro::GpuMemory<LbvhNode> d_bvhNodes;
		Oro::GpuMemory<u32> d_flags;
		u32 m_rootNodeIdx = 0;
		Timer m_timer;
		u32 m_nInternalNodes = 0;
		float m_cost = 0.0f;
	};
}