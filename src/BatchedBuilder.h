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
	struct BatchedBuildInput
	{
		std::vector<Triangle> m_primitives;
	};

	class BatchedBvhBuilder
	{
	public:
		void build(Context& context, std::vector<BatchedBuildInput>& batch);
		
		void traverseBvh(Context& context);

		Oro::GpuMemory<Bvh2Node> d_bvhNodes;
		Oro::GpuMemory<PrimRef> d_primRefs;
		Oro::GpuMemory<u32> d_rootNodes;
		u32 m_rootNodeIdx = 0;
		Timer m_timer;
		u32 m_nInternalNodes = 0;
		float m_cost = 0.0f;
	};
}