#include "Bvh.h"

using namespace BvhConstruction;

void LBVHBuild(std::vector<Triangle>& primitives)
{

}

void SBVHBuild(std::vector<Triangle>& primitives)
{

}

void Bvh::build(std::vector<Triangle>& primitives, const BvhBuildType buildType)
{
	if (buildType == BvhBuildType::BvhBuiltTypeLBVH)
	{
		LBVHBuild(primitives);
	}
	else if (buildType == BvhBuildType::BvhBuiltTypeSBVH)
	{
		SBVHBuild(primitives);
	}
}