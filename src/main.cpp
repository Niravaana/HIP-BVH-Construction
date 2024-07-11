#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <src/Error.h>
#include <src/Kernel.h>
#include <src/Timer.h>

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


using u32 = unsigned int;
using u64 = unsigned long;
enum Error
{
	Success,
	Failure
};

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

int main(int argc, char* argv[])
{
	try
	{
		oroDevice	orochiDevice;
		oroCtx		orochiCtxt;
		Timer timer;
		enum {
			reductionTime = 0
		};

		CHECK_ORO((oroError)oroInitialize((oroApi)(ORO_API_HIP), 0));
		CHECK_ORO(oroInit(0));
		CHECK_ORO(oroDeviceGet(&orochiDevice, 0)); // deviceId should be taken from user?
		CHECK_ORO(oroCtxCreate(&orochiCtxt, 0, orochiDevice));

		
		CHECK_ORO(oroCtxDestroy(orochiCtxt));
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}