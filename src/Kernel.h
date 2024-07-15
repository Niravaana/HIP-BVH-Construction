#pragma once
#include <dependencies/Orochi/Orochi/Orochi.h>
#include "Common.h"
#include <vector>
#include <filesystem>
#include <optional>

namespace BvhConstruction
{
	class Kernel
	{
	
	public:
		struct Argument
		{
			int			m_size;
			int			m_align;
			const void* m_value;

			template <class T>
			Argument(const T& value)
			{
				m_size = sizeof(T);
				m_align = __alignof(T);
				m_value = &value;
			}
		};

		Kernel(oroFunction function = 0) : m_function(function) {}

		void setArgs(std::vector<Argument> args);

		void launch(int gx, int gy, int gz, int bx, int by, int bz, u32 sharedMemBytes = 0, oroStream stream = 0);

		void launch(int nx, oroStream stream = 0, u32 sharedMemBytes = 0);
		void launch(int nx, int tx, oroStream stream = 0, u32 sharedMemBytes = 0);

		void setFunction(oroFunction function) { m_function = function; }
		oroFunction getFunction() { return m_function; }

		int getNumSmem();
		int getNumRegs();

	private:
		oroFunction		   m_function;
		std::vector<u8>	   m_args;
		std::vector<void*> m_argPtrs;
	};

	Error buildKernelFromSrc(Kernel& kernel, oroDevice& device, const std::filesystem::path& srcPath, const std::string& functionName, std::optional<std::vector<const char*>> opts);
}; 
