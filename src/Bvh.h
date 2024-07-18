#pragma once
#include "Kernel.h"
#include <vector>

namespace BvhConstruction
{
	enum class BvhBuildType
	{
		BvhBuiltTypeLBVH,
		BvhBuiltTypeSBVH,
	};

	class Bvh
	{
	public:
		void build(std::vector<Triangle>& primitives, const BvhBuildType buildType);
	};
}