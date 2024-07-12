
function copydir(src_dir, dst_dir, filter, single_dst_dir)
    filter = filter or "**"
    src_dir = src_dir .. "/"
    print('copy "' .. src_dir .. filter .. '" to "' .. dst_dir .. '".')
    dst_dir = dst_dir .. "/"
    local dir = path.rebase(".", path.getabsolute("."), src_dir) -- root dir, relative from src_dir

    os.chdir(src_dir) -- change current directory to src_dir
    local matches = os.matchfiles(filter)
    os.chdir(dir) -- change current directory back to root

    local counter = 0
    for k, v in ipairs(matches) do
        local target = iif(single_dst_dir, path.getname(v), v)
        --make sure, that directory exists or os.copyfile() fails
        os.mkdir(path.getdirectory(dst_dir .. target))
        if os.copyfile(src_dir .. v, dst_dir .. target) then
            counter = counter + 1
        end
    end

    if counter == #matches then
        print(counter .. " files copied.")
        return true
    else
        print("Error: " .. counter .. "/" .. #matches .. " files copied.")
        return nil
    end
end

workspace "BvhConstruction"
    configurations {"Debug", "Release", "RelWithDebInfo", "DebugGpu" }
    language "C++"
    platforms "x64"
    architecture "x86_64"

	if os.ishost("windows") then
		defines {"__WINDOWS__"}
	end
    characterset("MBCS")

    filter {"platforms:x64", "configurations:Debug or configurations:DebugGpu"}
      targetsuffix "64D"
      defines {"DEBUG"}
      symbols "On"
    filter {"platforms:x64", "configurations:DebugGpu"}
      defines {"DEBUG_GPU"}
    filter {"platforms:x64", "configurations:Release or configurations:RelWithDebInfo"}
      targetsuffix "64"
      defines {"NDEBUG"}
      optimize "On"
    filter {"platforms:x64", "configurations:RelWithDebInfo"}
      symbols "On"
    filter {}
	flags { "MultiProcessorCompile" }

    if os.ishost("windows") then
        buildoptions {"/wd4244", "/wd4305", "/wd4018", "/wd4996"}
    end
    if os.ishost("linux") then
        buildoptions {"-fvisibility=hidden"}
        buildoptions {"-fpermissive"}
    end
    defines {"__USE_HIP__"}

    targetdir "dist/bin/%{cfg.buildcfg}"    
    location "build/"
    
    project( "BvhConstruction" )
        cppdialect "C++17"
        kind "ConsoleApp"
    	    
		if os.istarget("windows") then
			links{ "version" }
		end
			
		sysincludedirs {"./","./dependencies/Orochi/" }
		
		files {"src/**.h", "src/**.cpp"}
		files {"dependencies/Orochi/Orochi/**.h", "dependencies/Orochi/Orochi/**.cpp"}
		files {"dependencies/Orochi/contrib/**.h", "dependencies/Orochi/contrib/**.cpp"}
		if os.istarget("windows") then
			copydir("./dependencies/Orochi/contrib/bin/win64", "./dist/bin/debug/")
			copydir("./dependencies/Orochi/contrib/bin/win64", "./dist/bin/release/")
			copydir("./dependencies/Orochi/contrib/bin/win64", "./dist/bin/RelWithDebInfo/")
		end
	