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

### Windows with VSCode

#### Required Software
- VSCode with extensions:
  - C/C++ (Microsoft)
  - CMake Tools (Microsoft)
  - CMake (twxs) - optional for syntax highlighting
- Visual Studio Community 2022 (free IDE with CUDA support)
- NVIDIA CUDA Toolkit for Windows
- vcpkg package manager

*Note: NVIDIA requires Visual Studio (not just Build Tools) for CUDA development on Windows. You can still use VSCode as your primary IDE while having Visual Studio installed for the CUDA toolchain.*

#### Installation Steps

1. **Install VSCode and Extensions**
   - Download VSCode from https://code.visualstudio.com/
   - Install the required extensions listed above

2. **Install Visual Studio Community 2022**
   - Download from: https://visualstudio.microsoft.com/vs/community/
   - Run the installer and select the "Desktop development with C++" workload
   - Also add this Individual Component: "C++ Clang Compiler for Windows (19.1.5)"
   - Ensure these components are included:
     - MSVC v143 compiler toolset
     - Windows 11 SDK (or Windows 10 SDK)
     - CMake tools for Visual Studio
     - C++ Clang Compiler for Windows (for clang-format)

3. **Install NVIDIA CUDA Toolkit**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose Windows → x86_64 → your Windows version → exe (network or local)
   - Run the installer with default settings (~3GB download)
   - Verify installation: Open Command Prompt and run `nvcc --version`

4. **Setup vcpkg Package Manager**
   - Clone and bootstrap vcpkg:
     ```cmd
     git clone https://github.com/Microsoft/vcpkg.git
     cd vcpkg
     .\bootstrap-vcpkg.bat
     ```

   - Install required dependencies:
     ```cmd
     .\vcpkg install curl:x64-windows openssl:x64-windows nlohmann-json:x64-windows gtest:x64-windows
     ```

   - Integrate vcpkg with CMake:
     ```cmd
     .\vcpkg integrate install
     ```
     This command will output a CMAKE_TOOLCHAIN_FILE path that you can use if integration doesn't work automatically.

## Building

### VSCode Development
1. Open the `find-optimal` folder in VSCode
2. The CMake Tools extension should detect the CMakeLists.txt automatically
3. Configure VSCode CMake settings (Press `Ctrl + ,` to open settings):
   - Search for "cmake generator" and set **Cmake: Generator** to `Ninja`
   - Search for "cmake configure args" and add to **Cmake: Configure Args**:
     ```
     -DCMAKE_TOOLCHAIN_FILE=C:/path/to/your/vcpkg/scripts/buildsystems/vcpkg.cmake
     ```
     (Replace with your actual vcpkg installation path)
   - Search for "cmake ctest args" and set **Cmake: Ctest Args** to `--verbose` for detailed test output
4. Configure the project: Press `Ctrl+Shift+P` → "CMake: Configure"
5. When prompted to select a kit:
   - Choose **[Scan for kits]** first to detect all available compilers
   - Select **"Visual Studio Community 2022 Release - amd64"** (or similar Visual Studio option)
   - Note: "amd64" is correct for 64-bit systems (same as x64)
6. Build: Press `Ctrl+Shift+P` → "CMake: Build" or use the build button in the status bar

#### Running and Debugging
To run the program with arguments and environment variables:

1. Create `.vscode/launch.json` in your project root:
   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "find-optimal",
               "type": "cppvsdbg",
               "request": "launch",
               "program": "${workspaceFolder}/build/find-optimal.exe",
               "args": ["--frame-mode", "random", "--verbose", "--log-file", "${workspaceFolder}/path/to/logfile.log"],
               "cwd": "${workspaceFolder}",
               "environment": [
                   {
                       "name": "GOOGLE_WEBAPP_URL",
                       "value": "https://your-webapp-url.com"
                   },
                   {
                       "name": "GOOGLE_API_KEY",
                       "value": "your-api-key-here"
                   }
               ],
               "console": "integratedTerminal"
           }
       ]
   }
   ```

2. Run with `F5` (debug) or `Ctrl+F5` (run without debugging)

### Visual Studio IDE Development
1. Open Visual Studio Community 2022
2. Choose "Open a local folder" and select the `find-optimal` directory
3. Visual Studio will automatically detect the CMakeLists.txt and configure the project
4. Select your build configuration (Debug/Release) from the toolbar
5. Build using `Ctrl+Shift+B` or Build → Build All

### Command Line Build
```bash
# Linux/WSL
cmake -B build && cmake --build build -j$(nproc)

# Windows
cmake -B build && cmake --build build
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