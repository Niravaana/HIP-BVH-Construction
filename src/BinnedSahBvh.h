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

		void traverseBvh(Context& context, std::vector<Triangle>& primitives, u32 nTotalNodes);

		std::vector<SahBvhNode> m_bvhNodes;
		Oro::GpuMemory<SahBvhNode> d_bvhNodes;
	};
}