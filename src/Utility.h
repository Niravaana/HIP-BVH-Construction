#pragma once
#include <src/Common.h>
#include <vector>

namespace BvhConstruction
{
	class Utility
	{
	public:
		static bool checkLbvhRootAabb(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);
		
		static bool checkLBvhCorrectness(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);

		static bool checkLBvh4Correctness(const Bvh4Node* bvhNodes, const PrimNode* wideLeafNodes, u32 rootIdx, u32 nInternalNodes);
		
		static bool checkSahCorrectness(const SahBvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes);
		
		static void TraversalLbvhCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height, u32 nInternalNodes);
		
		static void TraversalSahBvhCPU(const std::vector<Ray>& rayBuff, std::vector<SahBvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height);
		
		static float calculateLbvhCost(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);
	
		static float calculateBinnedSahBvhCost(const SahBvhNode* bvhNodes, u32 rootIdx, u32 totalNodes);

		static void generateTraversalHeatMap(std::vector<u32> rayCounter, u32 width, u32 height);
	};
}