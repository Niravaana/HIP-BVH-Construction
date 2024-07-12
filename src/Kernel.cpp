#include <src/Kernel.h>
#include <src/Error.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <optional>
#include <cstdlib> 
#include <ctime>
#include <cassert>

namespace BvhConstruction
{
	static std::string readSourceCode(const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>*> includes)
	{
		std::string	  src;
		std::ifstream file(path);
		if (!file.is_open())
		{
			std::string msg = "Unable to open " + path.string();
			throw std::runtime_error(msg);
		}
		size_t sizeFile;
		file.seekg(0, std::ifstream::end);
		size_t size = sizeFile = static_cast<size_t>(file.tellg());
		file.seekg(0, std::ifstream::beg);
		if (includes.has_value())
		{
			std::string line;
			while (std::getline(file, line))
			{
				if (line.find("#include") != std::string::npos)
				{
					size_t		pa = line.find("<");
					size_t		pb = line.find(">");
					std::string buf = line.substr(pa + 1, pb - pa - 1);
					includes.value()->push_back(buf);
					src += line + '\n';
				}
				src += line + '\n';
			}
		}
		else
		{
			src.resize(size, ' ');
			file.read(&src[0], size);
		}
		return src;
	}

	Error buildKernelFromSrc(Kernel& kernel, oroDevice& device, const std::filesystem::path& srcPath, const std::string& functionName, std::optional<std::vector<const char*>> opts)
	{
		oroFunction function = nullptr;
		std::vector<char> codec;
		std::vector<const char*> options;
		if (opts)
		{
			options = *opts;
		}
#if defined(HLT_DEBUG_GPU)
		options.push_back("-G");
#endif
		const bool isAmd = oroGetCurAPI(0) == ORO_API_HIP;
		std::string sarg;
		if (isAmd)
		{
			oroDeviceProp props;
			oroGetDeviceProperties(&props, device);
			sarg = std::string("--gpu-architecture=") + props.gcnArchName;
			options.push_back(sarg.c_str());
		}
		else
		{
			options.push_back("--device-c");
			options.push_back("-arch=compute_60");
		}
		options.push_back("-I../dependencies/Orochi/");
		options.push_back("-I../");
		options.push_back("-std=c++17");

		std::vector<std::filesystem::path> includeNamesData;
		std::string srcCode = readSourceCode(srcPath, &includeNamesData);
		if (srcCode.empty())
		{
			std::cerr << "Unable to open '" + srcPath.string() + "'" + "\n";
			return Error::Failure;
		}
		orortcProgram prog = nullptr;
		CHECK_ORORTC(orortcCreateProgram(&prog, srcCode.data(), functionName.c_str(), 0, nullptr, nullptr));

		orortcResult e = orortcCompileProgram(prog, static_cast<int>(options.size()), options.data());
		if (e != ORORTC_SUCCESS)
		{
			size_t logSize;
			CHECK_ORORTC(orortcGetProgramLogSize(prog, &logSize));

			if (logSize)
			{
				std::string log(logSize, '\0');
				CHECK_ORORTC(orortcGetProgramLog(prog, &log[0]));
				std::cerr << log;
				return Error::Failure;
			}
		}

		size_t codeSize = 0;
		orortcGetCodeSize(prog, &codeSize);

		codec.resize(codeSize);
		orortcGetCode(prog, codec.data());

		orortcDestroyProgram(&prog);

		oroModule module = nullptr;
		oroModuleLoadData(&module, codec.data());
		oroModuleGetFunction(&function, module, functionName.c_str());

		kernel.setFunction(function);

		return Error::Success;
	}


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
