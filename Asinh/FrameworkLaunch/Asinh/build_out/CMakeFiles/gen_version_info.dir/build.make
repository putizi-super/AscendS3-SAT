# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out

# Utility rule file for gen_version_info.

# Include any custom commands dependencies for this target.
include CMakeFiles/gen_version_info.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gen_version_info.dir/progress.make

CMakeFiles/gen_version_info:
	bash /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/cmake/util/gen_version_info.sh /usr/local/Ascend/ascend-toolkit/latest /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out

gen_version_info: CMakeFiles/gen_version_info
gen_version_info: CMakeFiles/gen_version_info.dir/build.make
.PHONY : gen_version_info

# Rule to build all files generated by this target.
CMakeFiles/gen_version_info.dir/build: gen_version_info
.PHONY : CMakeFiles/gen_version_info.dir/build

CMakeFiles/gen_version_info.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gen_version_info.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gen_version_info.dir/clean

CMakeFiles/gen_version_info.dir/depend:
	cd /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out /root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out/CMakeFiles/gen_version_info.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gen_version_info.dir/depend

