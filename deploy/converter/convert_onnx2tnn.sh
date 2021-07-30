#!/bin/bash

if [ $# != 2 ]; then
    echo "$0 <onnx_path> <output_dir>"
    exit 0
fi

if [ -z $TNN_ROOT ]; then
    echo "Please set env [TNN_ROOT]";
    exit -1;
fi

tnn_converter_dir=${TNN_ROOT}tools/convert2tnn/
onnx_path=$(readlink -f "$1")
echo $onnx_path

output_dir=$(readlink -f "$2")
echo $output_dir
cd $tnn_converter_dir
python converter.py onnx2tnn  $onnx_path -in input:1,3,112,112 -o $output_dir 
