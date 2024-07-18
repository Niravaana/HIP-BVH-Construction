#include <src/Error.h>
#include <iostream>
#include "Context.h"

using namespace BvhConstruction;

Context::Context()
{
	CHECK_ORO((oroError)oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0));
	CHECK_ORO(oroInit(0));
	CHECK_ORO(oroDeviceGet(&m_orochiDevice, 0)); // deviceId should be taken from user?
	CHECK_ORO(oroCtxCreate(&m_orochiCtxt, 0, m_orochiDevice));
	oroDeviceProp props;
	CHECK_ORO(oroGetDeviceProperties(&props, m_orochiDevice));
	std::cout << "Executing on '" << props.name << "'" << std::endl;
}

BvhConstruction::Context::~Context()
{
	CHECK_ORO(oroCtxDestroy(m_orochiCtxt));
}


