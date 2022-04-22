#!/bin/bash
echo '==================================================='
echo 'Installing necessary packages'
sudo apt-get update
sudo apt-get install -y gcc libtinfo-dev zlib1g-dev \
    build-essential cmake libedit-dev libxml2-dev

echo '==================================================='
echo 'Downloading llvm_8.0'
wget https://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
mv clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04 clang+llvm-8.0.0

echo '==================================================='
echo 'Installing TVM'
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
sed -i "s|USE_LLVM OFF|USE_LLVM $(cd ../../; pwd)/clang+llvm-8.0.0/bin/llvm-config|" config.cmake
sed -i "s|USE_CUDA OFF|USE_CUDA ON|" config.cmake
cmake ..
make -j 8

echo '==================================================='
echo 'Finish!'