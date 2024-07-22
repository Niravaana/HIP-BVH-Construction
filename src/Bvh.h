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
	enum TimerCodes
	{
		CalculateCentroidExtentsTime,
		CalculateMortonCodesTime,
		SortingTime,
		BvhBuildTime,
		TraversalTime
	};

	class LBVH
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
		Oro::GpuMemory<LbvhNode> d_bvhNodes;
		Oro::GpuMemory<u32> d_flags;
		u32 m_rootNodeIdx = INVALID_NODE_IDX;
		Timer m_timer;
	};

	struct PrimitveRef
	{
		Aabb m_aabb; //world space aabb
		size_t m_primId;
	};

	struct Task
	{
		u32 m_nodeIdx;
		u32 m_start;
		u32 m_end;
	};

	class SahBvh
	{
	public:
		void build(Context& context, std::vector<Triangle>& primitives);

		void traverseBvh(Context& context);

		std::vector<SahBvhNode> m_bvhNodes;
		Oro::GpuMemory<SahBvhNode> d_bvhNodes;
	};

	class HPloc
	{
	public:
		void build(Context& context, std::vector<Triangle>& primitives);

		void traverseBvh(Context& context);
	};
}