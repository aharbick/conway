# Build Instructions

This project uses CMake for cross-platform builds and supports both Ubuntu/WSL and Windows environments.

## Prerequisites

### Ubuntu/WSL (Recommended)
Run the setup script to install all dependencies:
```bash
./bin/setup.sh
```

This installs:
- Build tools (cmake, build-essential, pkgconf)
- CUDA Toolkit (commented instructions available for manual install)
- Development libraries (libcurl, libssl, gtest)
- Code formatting tools (clang-format)
- Additional utilities (direnv, nlohmann-json)

### Windows
- Visual Studio 2019/2022 with C++ workload
- NVIDIA CUDA Toolkit for Windows
- CMake 3.18+

## Building

### Standard Build
```bash
cmake -B build && cmake --build build -j$(nproc)
```

### Windows Build for Visual Profiler
```cmd
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release
```

## Available Targets

After the initial build, you can run specific targets from the build directory:
```bash
cd build
make run-tests      # Run unit tests
make format-check   # Check code formatting
make format-fix     # Auto-format code
make clean          # Clean build artifacts
```

Or run them directly with cmake:
```bash
cmake --build build --target run-tests     # Run unit tests
cmake --build build --target format-check  # Check code formatting
cmake --build build --target format-fix    # Auto-format code
rm -rf build                               # Clean everything
```

## Executables

After building, you'll find these executables in the `build/` directory:
- `find-optimal` - Main Conway's Game of Life optimizer
- `explore-cachability` - Cache analysis utility
- `validate-frame-search-completeness` - Search validation tool
- `run_tests` - Unit test suite

## GPU Architecture Detection

The build system automatically detects your GPU and sets the appropriate CUDA architecture:
- RTX 5090/4090: compute capability 12.0/8.9 → sm_89
- RTX 3090: compute capability 8.6 → sm_86
- RTX 3080: compute capability 8.6 → sm_86
- GTX 1080: compute capability 7.5 → sm_75

## NVIDIA Visual Profiler

For optimal profiling experience:
1. Use the Windows CMake build with Visual Studio generator
2. Build in Release configuration
3. The executable will be compatible with NVIDIA Visual Profiler/Nsight Compute