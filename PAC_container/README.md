# PAC description

## Stream_16x64
Type: Full kernel

Model size:
- Input Population, Hypercolumn 784, Minicolumn 2, input image 28x28
- Hidden Population, Hypercolumn 16, Minicolumn 64, Active Neuron 16, Silent Neuron 16
- Output Population, Hypercolumn 1, Minicolumn 10, Classification 10 classes 

Data type: float 32-bit

Target dataset: MNIST

## stream_32x128_SP
Type: Full kernel

Model size:
- Input Population, Hypercolumn 784, Minicolumn 2, input image 28x28
- Hidden Population, Hypercolumn 32, Minicolumn 128, Active Neuron 64, Silent Neuron 64
- Output Population, Hypercolumn 1, Minicolumn 10, Classification 10 classes 

Data type: float 32-bit

Target dataset: MNIST

## infer_fixed_32x128

Type: Inference kernel

Model size:
- Input Population, Hypercolumn 784, Minicolumn 2, input image 28x28
- Hidden Population, Hypercolumn 32, Minicolumn 128, Active Neuron 64, Silent Neuron 64
- Output Population, Hypercolumn 1, Minicolumn 10, Classification 10 classes 

Data type: Fixed point size 20-bit with 6-bits integer

Target dataset: MNIST