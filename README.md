# Demo Video Inference for BCPNN

This repository demonstrates running Bayesian Confidence Propagation Neural Network (BCPNN) inference on video input using a Xilinx FPGA platform.

## Install PAC

For this video inference example, install the PAC named **mnist_float**.

## Compile the application

Use the cross-compiler for aarch64 to build the host application:

```bash
make host CXX=/usr/bin/aarch64-linux-gnu-g++ HOST_ARCH=aarch64 EDGE_COMMON_SW=/home/ubuntu HOST_COMPILE=mnistmain_video_new
```

## Run the application

The executable expects five arguments:

```
<exe_file> <par_file> <xclbin> <weight_bin> <video_file>
```

Example invocation:

```bash
./mnistmain_video_new \
  ./test/MNIST_ZCU104/mnistmain.par \
  ./PAC_container/hwconfig/mnist_float/zcu104/BCPNN_infer_float.xclbin \
  ./TrainedWeight/alvis_fullmnist_32x128_64x64_eps-4.bin \
  ./video_input/output.avi
```
