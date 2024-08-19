#pragma once
#include<dependencies/Orochi/Orochi/Orochi.h>
#include <src/Common.h>

namespace BvhConstruction
{
	class Context
	{
	public:
		Context();
		~Context();

		u32 getMaxGridSize() const;

		oroDevice	m_orochiDevice;
		oroCtx		m_orochiCtxt;
		oroDeviceProp m_devProp;
	};
}