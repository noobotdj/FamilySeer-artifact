# FamilySeer-artifact
Evaluation code of FamilySeer

# Install TVM
Clone the repo.

```
git clone --recursive https://github.com/noobotdj/FamilySeer-artifact.git
cd FamilySeer-artifact
```

Run `build_tvm.sh` to install TVM.

```
./build_tvm.sh
```
TVM requires python and llvm. Please follow the command line below if you have trouble running `build_tvm.sh`.

```
cd tvm
mkdir build
cp cmake/config.cmake build
```
Change `set(USE_CUDA OFF)` to `set(USE_CUDA ON)` to enable CUDA backend, change `set(USE_LLVM OFF)` to `set(USE_LLVM ("Your 'llvm-config' location"))` to enable CPU codegen.

```
cd build
cmake ..
make -j 8
```

If no errors occur, you can go to the next step. 
If not, please refer to [TVM installing from Source](https://tvm.apache.org/docs/install/from_source.html).

# Set environment 
Append the following line in `~/.bashrc`
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
# Run test
Please go to `./scripts/`

# Tested platforms
All platforms are running in Ubuntu 20.04, GCC 9.3 and LLVM-8.0
- Intel Xeon Silver 4210 CPU (Skylake, with `avx512`)
- NVIDIA V100 GPU (Volta)

