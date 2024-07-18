#pragma once
#include<dependencies/Orochi/Orochi/Orochi.h>

namespace BvhConstruction
{
	class Context
	{
	public:
		Context();
		~Context();
		oroDevice	m_orochiDevice;
		oroCtx		m_orochiCtxt;
	};
}