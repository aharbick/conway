cmake_minimum_required(VERSION 3.18)

# Auto-detect GPU architecture BEFORE project declaration to avoid caching issues
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Map compute capability to supported architecture and force cache override
if(GPU_COMPUTE_CAP STREQUAL "12.0")
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)
elseif(GPU_COMPUTE_CAP STREQUAL "8.9")
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)
elseif(GPU_COMPUTE_CAP STREQUAL "8.7")
    set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures" FORCE)
elseif(GPU_COMPUTE_CAP STREQUAL "8.6")
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures" FORCE)
elseif(GPU_COMPUTE_CAP STREQUAL "7.5")
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures" FORCE)
else()
    set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures" FORCE)  # Default fallback
endif()

message(STATUS "Pre-project GPU detection: ${GPU_COMPUTE_CAP} -> sm_${CMAKE_CUDA_ARCHITECTURES}")

project(find-optimal LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Print helpful information for Windows users
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "")
    message(STATUS "Windows build detected. If you encounter dependency errors:")
    message(STATUS "1. Install vcpkg dependencies: vcpkg install curl:x64-windows openssl:x64-windows nlohmann-json:x64-windows")
    message(STATUS "2. Run vcpkg integration: vcpkg integrate install")
    message(STATUS "3. Or set CMAKE_TOOLCHAIN_FILE to your vcpkg toolchain file")
    message(STATUS "")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Handle argp library for cross-platform support
if(WIN32)
    include(FetchContent)
    FetchContent_Declare(
        argp-standalone
        GIT_REPOSITORY https://github.com/tom42/argp-standalone.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(argp-standalone)
    set(ARGP_LIBRARIES argp-standalone)
else()
    # On Unix systems, argp is typically part of glibc
    set(ARGP_LIBRARIES "")
endif()

# Handle libcurl and OpenSSL dependencies
find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    if(WIN32)
        message(FATAL_ERROR "libcurl not found. Please install via vcpkg: vcpkg install curl:x64-windows")
    else()
        message(FATAL_ERROR "libcurl not found. Please install libcurl development package.")
    endif()
endif()

find_package(OpenSSL REQUIRED)
if(NOT OpenSSL_FOUND)
    if(WIN32)
        message(FATAL_ERROR "OpenSSL not found. Please install via vcpkg: vcpkg install openssl:x64-windows")
    else()
        message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL development package.")
    endif()
endif()

# Find nlohmann_json (used by google_client.h)
find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    if(WIN32)
        message(FATAL_ERROR "nlohmann_json not found. Please install via vcpkg: vcpkg install nlohmann-json:x64-windows")
    else()
        message(FATAL_ERROR "nlohmann_json not found. Please install nlohmann-json development package.")
    endif()
endif()

if(WIN32)
    # On Windows with vcpkg, use the imported targets
    set(CURL_LIBRARIES CURL::libcurl)
    set(OPENSSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)
else()
    # On Unix systems, use the imported targets
    set(CURL_LIBRARIES CURL::libcurl)
    set(OPENSSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)
endif()

# GPU architecture was already detected and set before project() declaration

# Include directories
include_directories(include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# CPU-only source files
set(CPU_SOURCES
    src/api_test_handlers.cpp
    src/cli_parser.cpp
    src/utils.cpp
    src/logging.cpp
)

# CUDA source files (including .cpp files that use CUDA)
set(CUDA_SOURCES
    src/gol_cuda.cu
    src/gol.cpp
    src/gol_search.cpp
    src/simulation_handlers.cpp
    src/subgrid_cache.cpp
    src/find-optimal.cpp
)

# Create executable
add_executable(find-optimal ${CPU_SOURCES} ${CUDA_SOURCES})

# Set CUDA properties for CUDA source files
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Link libraries
target_link_libraries(find-optimal
    CUDA::cudart
    CUDA::cuda_driver
    ${ARGP_LIBRARIES}
    ${CURL_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(find-optimal ws2_32 crypt32)
else()
    # Check if we're in WSL
    execute_process(
        COMMAND grep -q microsoft /proc/version
        RESULT_VARIABLE IS_WSL
        ERROR_QUIET
    )

    if(IS_WSL EQUAL 0)
        # WSL-specific library paths
        target_link_directories(find-optimal PRIVATE /usr/lib/wsl/lib)
    endif()

    target_link_libraries(find-optimal pthread)
endif()

# Set CUDA properties
set_target_properties(find-optimal PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable extended lambda for CUDA (required for device lambdas)
target_compile_options(find-optimal PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Additional executables
add_executable(explore-subgrid src/explore-subgrid.cpp)
add_executable(validate-frame-search-completeness src/validate_frame_search_completeness.cpp)

# Code formatting targets
find_program(CLANG_FORMAT_EXE NAMES "clang-format")
if(CLANG_FORMAT_EXE)
    # Collect all source files for formatting
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/tests/*.cpp
    )

    # Format check target
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${ALL_SOURCE_FILES}
        COMMENT "Checking if code formatting is needed..."
        VERBATIM
    )

    # Format fix target
    add_custom_target(format-fix
        COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
        COMMENT "Formatting code..."
        VERBATIM
    )
else()
    message(WARNING "clang-format not found - formatting targets will not be available")
endif()

# Unit testing support
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()

    # Find required packages for testing
    if(WIN32)
        # On Windows with vcpkg, find GTest via CONFIG mode
        find_package(GTest CONFIG)
        set(GTEST_FOUND ${GTest_FOUND})
        if(GTEST_FOUND)
            set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
            set(GTEST_INCLUDE_DIRS "")
        endif()
    else()
        # On Linux/WSL, use PkgConfig for both gtest and gtest_main
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GTEST REQUIRED gtest)
        pkg_check_modules(GTEST_MAIN REQUIRED gtest_main)

        # Combine the libraries
        set(GTEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
        set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS} ${GTEST_MAIN_INCLUDE_DIRS})
    endif()

    if(GTEST_FOUND)
        # Collect test source files
        file(GLOB TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/*.cpp)

        if(TEST_SOURCES)
            # Test-specific source files (non-CUDA sources needed for tests)
            set(TEST_IMPL_SOURCES
                src/utils.cpp
                src/logging.cpp
            )

            # Create test executable with both test sources and implementation sources
            add_executable(run_tests ${TEST_SOURCES} ${TEST_IMPL_SOURCES})

            # Link test libraries
            target_link_libraries(run_tests
                ${GTEST_LIBRARIES}
                ${CURL_LIBRARIES}
                ${OPENSSL_LIBRARIES}
            )

            # Platform-specific additional libraries
            if(NOT WIN32)
                target_link_libraries(run_tests pthread)
            endif()

            # Include test directories
            target_include_directories(run_tests PRIVATE
                ${CMAKE_SOURCE_DIR}/include
                ${GTEST_INCLUDE_DIRS}
            )

            # Add test compilation flags
            target_compile_options(run_tests PRIVATE ${GTEST_CFLAGS})

            # Register the test
            add_test(NAME unit_tests COMMAND run_tests)

            # Custom test target
            add_custom_target(run-tests
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
                DEPENDS run_tests
                COMMENT "Running unit tests..."
            )
        else()
            message(STATUS "No test files found in tests/ directory")
        endif()
    else()
        message(WARNING "gtest not found - unit testing will not be available")
    endif()
endif()
