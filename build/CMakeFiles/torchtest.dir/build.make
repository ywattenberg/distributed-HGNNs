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
CMAKE_SOURCE_DIR = /home/zkoutsok2/distributed-THNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zkoutsok2/distributed-THNN/build

# Include any dependencies generated for this target.
include CMakeFiles/torchtest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torchtest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torchtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torchtest.dir/flags.make

CMakeFiles/torchtest.dir/main.cpp.o: CMakeFiles/torchtest.dir/flags.make
CMakeFiles/torchtest.dir/main.cpp.o: ../main.cpp
CMakeFiles/torchtest.dir/main.cpp.o: CMakeFiles/torchtest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zkoutsok2/distributed-THNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torchtest.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torchtest.dir/main.cpp.o -MF CMakeFiles/torchtest.dir/main.cpp.o.d -o CMakeFiles/torchtest.dir/main.cpp.o -c /home/zkoutsok2/distributed-THNN/main.cpp

CMakeFiles/torchtest.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchtest.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zkoutsok2/distributed-THNN/main.cpp > CMakeFiles/torchtest.dir/main.cpp.i

CMakeFiles/torchtest.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchtest.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zkoutsok2/distributed-THNN/main.cpp -o CMakeFiles/torchtest.dir/main.cpp.s

# Object files for target torchtest
torchtest_OBJECTS = \
"CMakeFiles/torchtest.dir/main.cpp.o"

# External object files for target torchtest
torchtest_EXTERNAL_OBJECTS =

torchtest: CMakeFiles/torchtest.dir/main.cpp.o
torchtest: CMakeFiles/torchtest.dir/build.make
torchtest: ../libs/libtorch/lib/libtorch.so
torchtest: ../libs/libtorch/lib/libc10.so
torchtest: ../libs/libtorch/lib/libkineto.a
torchtest: /usr/lib/x86_64-linux-gnu/libopenblas.so
torchtest: ../libs/libtorch/lib/libc10.so
torchtest: CMakeFiles/torchtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zkoutsok2/distributed-THNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable torchtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torchtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torchtest.dir/build: torchtest
.PHONY : CMakeFiles/torchtest.dir/build

CMakeFiles/torchtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torchtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torchtest.dir/clean

CMakeFiles/torchtest.dir/depend:
	cd /home/zkoutsok2/distributed-THNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zkoutsok2/distributed-THNN /home/zkoutsok2/distributed-THNN /home/zkoutsok2/distributed-THNN/build /home/zkoutsok2/distributed-THNN/build /home/zkoutsok2/distributed-THNN/build/CMakeFiles/torchtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torchtest.dir/depend
