## Deploy

### Introduction
After finishing the train, we deploy the model based on [TNN](https://github.com/Tencent/TNN) framework. TNN is a high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab, and supports for ARM CPU/X86 CPU/NV GPU/ devcies.

### Preparation

Download TNN source and compile core library and model-converter tools, please refer to the official guides, [Compile converter](https://github.com/Tencent/TNN/blob/master/doc/en/user/convert_en.md), [Compile TNN](https://github.com/Tencent/TNN/blob/master/doc/en/user/compile_en.md)

### Convert
1. Convert PyTorch checkpoint to ONNX model
```
python3 converter/export_onnx.py -h
```
The output shows
``` bash
usage: export_onnx.py [-h] --ckpt_path CKPT_PATH --onnx_name ONNX_NAME
                      --model_name MODEL_NAME

export pytorch model to onnx

optional arguments:
  -h, --help            show this help message and exit
  --ckpt_path CKPT_PATH
  --onnx_name ONNX_NAME
  --model_name MODEL_NAME
```

2. Convert ONNX model to TNN model

``` bash
./converter/convert_onnx2tnn.sh <onnx_path> <output_dir>
```

### Compile

1. Linux x86

Please compile TNN by using `build_linux_openvino.sh`. We suggest using the OpenVINO backend in x86 cpus for high performance and TNN integrates the OpenVINO interfaces.
Run below scripts to compile and test deploy codes.
```
./scripts/build_linux_x86.sh

./build/test -h
    -h     <help>
    -img   <input img>
    -proto <tnn proto path>
    -model <tnn model path>

```



