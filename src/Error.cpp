#include <src/Error.h>
#include <iostream>

using namespace GpuOptimizationProject;

namespace GpuOptimizationProject
{

	void checkOro(oroError res, const char* file, int line)
	{
		if (res != oroSuccess)
		{
			const char* msg;
			oroGetErrorString(res, &msg);
			std::cerr << "Orochi error %s  on line %d in file %s\n" << msg << line << file << std::endl;
		}
	}

	void checkOrortc(orortcResult res, const char* file, int line)
	{
		if (res != ORORTC_SUCCESS)
		{
			std::cerr << "Orortc error %s  on line %d in file %s\n" << orortcGetErrorString(res) << line << file << std::endl;
		}
	}
}