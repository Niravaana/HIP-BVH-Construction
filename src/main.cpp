#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <dependencies/Orochi/Orochi/GpuMemory.h>

#include <src/Error.h>
#include <src/Kernel.h>
#include <src/Timer.h>
#include <src/Context.h>
#include <src/Utility.h>
#include <iostream>

#include "SinglePassLbvh.h"
#include "TwoPassLbvh.h"
#include "BinnedSahBvh.h"
#include "PLOC++Bvh.h"
#include "Hploc.h"
#include "BatchedBuilder.h"

#define USE_TWOPASS_LBVH 1
//#define USE_SINGLEPASS_LBVH 1
//#define USE_BATCHED_BUILDER 1
//#define USE_PLOC 1
//#define USE_HPLOC 1 

using namespace BvhConstruction;

int main(int argc, char* argv[])
{
	try
	{
		Context context;
		Timer timer;

#if defined USE_BATCHED_BUILDER	
		{
			std::vector<Triangle> triangles;
			MeshLoader::loadScene("../src/meshes/cornellbox/cornellbox.obj", "../src/meshes/cornellbox/", triangles);

			BatchedBvhBuilder bvh;
			constexpr u32 batchSize = 2048 * 2;
			std::vector<BatchedBuildInput> batches(batchSize);

			for (size_t i = 0; i < batchSize; i++)
			{
				batches[i].m_primitives = triangles;
			}

			bvh.build(context, batches);
			bvh.traverseBvh(context);
		}
#endif

		std::vector<Triangle> triangles;
		MeshLoader::loadScene("../src/meshes/sponza/sponza.obj", "../src/meshes/sponza/", triangles);
	
#if defined USE_SINGLEPASS_LBVH
		SinglePassLbvh bvh;
		bvh.build(context, triangles);
		bvh.traverseBvh(context);
#endif 
	
#if defined USE_TWOPASS_LBVH
		TwoPassLbvh bvh;
		bvh.build(context, triangles);
		bvh.traverseBvh(context);
#endif
		
#if defined USE_PLOC
		PLOCNew bvh;
		bvh.build(context, triangles);
		bvh.traverseBvh(context);
#endif

#if defined USE_HPLOC
		HPLOC bvh;
		bvh.build(context, triangles);
		bvh.traverseBvh(context);
#endif

	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}