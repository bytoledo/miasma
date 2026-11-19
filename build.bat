@echo off
setlocal

set "VCVARS="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VCVARS%"=="" (
    echo [ERROR] No se encontro vcvars64.bat de Visual Studio / Build Tools.
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

if "%CUDA_PATH%"=="" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
)

if not exist build mkdir build

echo [1/4] Compilando src\main.cpp...
cl.exe /nologo /Zc:preprocessor /EHsc /std:c++17 /O2 /arch:AVX2 /openmp /W3 /Iinclude /I"%CUDA_PATH%\include" /c src\main.cpp /Fobuild\main.obj
if errorlevel 1 exit /b 1

echo [2/4] Compilando src\cpu_engine.cpp...
cl.exe /nologo /Zc:preprocessor /EHsc /std:c++17 /O2 /arch:AVX2 /openmp /W3 /Iinclude /c src\cpu_engine.cpp /Fobuild\cpu_engine.obj
if errorlevel 1 exit /b 1

echo [3/4] Compilando src\cuda_engine.cu (sm_120)...
nvcc -std=c++17 -arch=sm_120 -Xcompiler "/Zc:preprocessor /EHsc /O2 /arch:AVX2" -Iinclude -O3 --use_fast_math -c src\cuda_engine.cu -o build\cuda_engine.obj
if errorlevel 1 exit /b 1

echo [4/4] Enlazando build\seird_parallel.exe...
nvcc -std=c++17 -arch=sm_120 -Xcompiler "/Zc:preprocessor /EHsc /O2 /arch:AVX2 /openmp" build\main.obj build\cpu_engine.obj build\cuda_engine.obj -o build\seird_parallel.exe -lcurand -lopengl32 -luser32 -lgdi32 -lcomctl32
if errorlevel 1 exit /b 1

echo Compilacion exitosa: build\seird_parallel.exe
