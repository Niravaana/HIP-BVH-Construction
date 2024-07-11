#include <src/Kernel.h>
#include <src/Error.h>
#include <cstring>

namespace GpuOptimizationProject
{
	template <typename T, typename U>
	T divideRoundUp(T value, U factor)
	{
		return (value + factor - 1) / factor;
	}

	void Kernel::launch(int gx, int gy, int gz, int bx, int by, int bz, u32 sharedMemBytes, oroStream stream)
	{
		int tb, minNb;
		CHECK_ORO(oroModuleOccupancyMaxPotentialBlockSize(&minNb, &tb, m_function, 0, 0));
		CHECK_ORO(oroModuleLaunchKernel(m_function, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, m_argPtrs.data(), 0));
	}

	void Kernel::setArgs(std::vector<Argument> args)
	{
		int size = 0;
		for (int i = 0; i < args.size(); i++)
		{
			size = (size + args[i].m_align - 1) & -args[i].m_align;
			size += args[i].m_size;
		}

		m_args.clear();
		m_args.resize(size);
		m_argPtrs.clear();
		m_argPtrs.resize(size);

		int ofs = 0;
		for (int i = 0; i < args.size(); i++)
		{
			ofs = (ofs + args[i].m_align - 1) & -args[i].m_align;
			memcpy(m_args.data() + ofs, args[i].m_value, args[i].m_size);
			m_argPtrs[i] = m_args.data() + ofs;
			ofs += args[i].m_size;
		}
	}

	void Kernel::launch(int nx, oroStream stream, u32 sharedMemBytes)
	{
		int tb, minNb;
		CHECK_ORO(oroModuleOccupancyMaxPotentialBlockSize(&minNb, &tb, m_function, 0, 0));
		int nb = divideRoundUp(nx, tb);
		launch(nb, 1, 1, tb, 1, 1, sharedMemBytes, stream);
	}

	void Kernel::launch(int nx, int tx, oroStream stream, u32 sharedMemBytes)
	{
		int tb = tx;
		int nb = divideRoundUp(nx, tb);
		launch(nb, 1, 1, tb, 1, 1, sharedMemBytes, stream);
	}

	int Kernel::getNumSmem()
	{
		int numSmem;
		CHECK_ORO(oroFuncGetAttribute(&numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_function));
		return numSmem;
	}

	int Kernel::getNumRegs()
	{
		int numRegs;
		CHECK_ORO(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, m_function));
		return numRegs;
	}
}; 
