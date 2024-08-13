#pragma once
#include <src/Common.h>
#include <vector>
#include <string>

namespace BvhConstruction
{
	template <typename T, typename U>
	constexpr T divideRoundUp(T value, U factor)
	{
		return (value + factor - 1) / factor;
	}

	class MeshLoader
	{
	public:
		static void loadScene(const std::string& filename, const std::string& mtlBaseDir, std::vector<Triangle>& trianglesOut);
	};

	class Utility
	{
	public:
		static bool checkLbvhRootAabb(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);
		
		static bool checkLBvhCorrectness(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);

		static bool checkPlocBvhCorrectness(const LbvhNode* bvhNodes, const PrimRef* leafNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);

		static bool checkLBvh4Correctness(const Bvh4Node* bvhNodes, const PrimNode* wideLeafNodes, u32 rootIdx, u32 nInternalNodes);
		
		static bool checkSahCorrectness(const SahBvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes);
		
		static void TraversalLbvhCPU(const std::vector<Ray>& rayBuff, std::vector<LbvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height, u32 nInternalNodes);
		
		static void TraversalSahBvhCPU(const std::vector<Ray>& rayBuff, std::vector<SahBvhNode> bvhNodes, std::vector<Triangle> primitives, Transformation& t, u8* dst, u32 width, u32 height);
		
		static float calculateLbvhCost(const LbvhNode* bvhNodes, u32 rootIdx, u32 nLeafNodes, u32 nInternalNodes);

		static float calculatebvh4Cost(const Bvh4Node* bvhNodes, const LbvhNode* bvh2Nodes, u32 rootIdx, u32 totalNodes, u32 nInternalNodes);
	
		static float calculateBinnedSahBvhCost(const SahBvhNode* bvhNodes, u32 rootIdx, u32 totalNodes);

		static void generateTraversalHeatMap(std::vector<u32> rayCounter, u32 width, u32 height);

		static void doEarlySplitClipping(std::vector<Triangle>& inputPrims, std::vector<PrimRef>& outPrimRefs, float saMax = FltMax);

		static void collapseBvh2toBvh4(const std::vector<LbvhNode>& bvh2Nodes, std::vector<Bvh4Node>& bvh4Nodes, std::vector<PrimNode> bvh4LeafNodes, std::vector<uint2>& taskQ, u32 taskCount, u32& bvh8InternalNodeOffset, u32 nBvh2InternalNodes, u32 nBvh2LeafNodes);
		
	};
}
