# Demo Video Inference for BCPNN

## Install PAC 

for the inference video example the PAC is mnist_float. 

## Compile application

make host CXX=/usr/bin/aarch64-linux-gnu-g++ HOST_ARCH=aarch64 EDGE_COMMON_SW=/home/ubuntu HOST_COMPILE=mnistmain_video_new

## Run application

argument is

<exe_file> <par_file> <xclbin> <weight_bin> <video_file>

example is below

./mnistmain_video_new ./test/MNIST_ZCU104/mnistmain.par ./PAC_container/hwconfig/mnist_float/zcu104/BCPNN_infer_float.xclbin ./TrainedWeight/alvis_fullmnist_32x128_64x64_eps-4.bin ./video_input/output.avi

