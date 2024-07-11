#pragma once
#include<dependencies/Orochi/Orochi/Orochi.h>

namespace GpuOptimizationProject
{
	#define CHECK_ORO( error ) ( checkOro( error, __FILE__, __LINE__ ) )
	void checkOro( oroError res, const char* file, int line );
	
	#define CHECK_ORORTC( error ) ( checkOrortc( error, __FILE__, __LINE__ ) )
	void checkOrortc( orortcResult res, const char* file, int line );
}