@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
if errorlevel 1 exit /b 1

if not exist build mkdir build

echo [1/4] Compilando src\main.cpp...
cl.exe /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /openmp /W3 /Iinclude /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" /c src\main.cpp /Fobuild\main.obj
if errorlevel 1 exit /b 1

echo [2/4] Compilando src\cpu_engine.cpp...
cl.exe /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /openmp /W3 /Iinclude /c src\cpu_engine.cpp /Fobuild\cpu_engine.obj
if errorlevel 1 exit /b 1

echo [3/4] Compilando src\cuda_engine.cu (sm_75)...
nvcc -std=c++17 -arch=sm_75 -Iinclude -O3 --use_fast_math -c src\cuda_engine.cu -o build\cuda_engine.obj
if errorlevel 1 exit /b 1

echo [4/4] Enlazando build\seird_parallel.exe...
nvcc -std=c++17 -arch=sm_75 -Xcompiler "/EHsc /O2 /arch:AVX2 /openmp" build\main.obj build\cpu_engine.obj build\cuda_engine.obj -o build\seird_parallel.exe -lcurand -lopengl32 -luser32 -lgdi32 -lcomctl32
if errorlevel 1 exit /b 1

echo Compilacion exitosa: build\seird_parallel.exe
