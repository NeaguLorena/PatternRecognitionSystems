# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Lab_prs.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Lab_prs.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Lab_prs.dir/flags.make

CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o: CMakeFiles/Lab_prs.dir/flags.make
CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o: ../OpenCVApplication.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o -c /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/OpenCVApplication.cpp

CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/OpenCVApplication.cpp > CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.i

CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/OpenCVApplication.cpp -o CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.s

# Object files for target Lab_prs
Lab_prs_OBJECTS = \
"CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o"

# External object files for target Lab_prs
Lab_prs_EXTERNAL_OBJECTS =

Lab_prs: CMakeFiles/Lab_prs.dir/OpenCVApplication.cpp.o
Lab_prs: CMakeFiles/Lab_prs.dir/build.make
Lab_prs: /usr/local/lib/libopencv_dnn.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_highgui.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_ml.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_objdetect.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_shape.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_stitching.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_superres.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_videostab.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_calib3d.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_features2d.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_flann.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_photo.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_video.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_videoio.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_imgcodecs.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_imgproc.3.4.9.dylib
Lab_prs: /usr/local/lib/libopencv_core.3.4.9.dylib
Lab_prs: CMakeFiles/Lab_prs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Lab_prs"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Lab_prs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Lab_prs.dir/build: Lab_prs

.PHONY : CMakeFiles/Lab_prs.dir/build

CMakeFiles/Lab_prs.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Lab_prs.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Lab_prs.dir/clean

CMakeFiles/Lab_prs.dir/depend:
	cd /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug /Users/lorenaneagu/Year4Sem1/PRS/IP/OpenCVApplication-VS2017_OCV340_basic/cmake-build-debug/CMakeFiles/Lab_prs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Lab_prs.dir/depend

