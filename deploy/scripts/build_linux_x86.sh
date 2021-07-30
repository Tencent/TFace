#!/bin/bash

if [ -z $TNN_ROOT ]; then
    echo "Please set env [TNN_ROOT]";
    exit -1;
fi

if [ -z $OpenCV_DIR ]; then
    echo "Please set env [OpenCV_DIR]";
    exit -1;
fi

TNN_LIB_DIR=${TNN_ROOT}/scripts/x86_linux_release/lib/
TNN_INCLUDE_DIR=${TNN_ROOT}/scripts/x86_linux_release/include/
GFLAGS_DIR=${TNN_ROOT}/third_party/gflags/
OpenCV_DIR=${OpenCV_DIR}


if [ ! -d ${TNN_LIB_PATH} ]; then
    echo "TNN not build success, please build TNN first";
    exit -2;
fi

rm -rf ./build
mkdir build
cd build
cmake3 .. \
    -DTNN_LIB_DIR=${TNN_LIB_DIR} \
    -DTNN_INCLUDE_DIR=${TNN_INCLUDE_DIR} \
    -DOpenCV_DIR=${OpenCV_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR} \
    -DTNN_OPENVINO_ENABLE=ON \
    -DLOG_ENABLE=OFF
make
