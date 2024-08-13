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

using namespace BvhConstruction;

int main(int argc, char* argv[])
{
	try
	{
		Context context;
		Timer timer;
		TwoPassLbvh bvh;

		std::vector<Triangle> triangles;
		//MeshLoader::loadScene("../src/meshes/cornellbox/cornellbox.obj", "../src/meshes/cornellbox/", triangles);
		MeshLoader::loadScene("../src/meshes/bunny/bunny.obj", "../src/bunny/bunny/", triangles);

		bvh.build(context, triangles);
		bvh.traverseBvh(context);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}