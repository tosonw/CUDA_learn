# CUDA_learn
我用来学习CUDA的程序，内部有多个demo，只依赖CUDA，可以自己使用nvcc去编译。
由于我使用的是Geany编译器，它内部直接编译运行，所以本程序没有Makefile。

依赖：
CUDA（我是在CUDA-9.0环境上学习的）


编译：
（我使用Geany编译器可以直接编译运行，这里我还是写出nvcc编译命令，如下：）
```shell
 (in directory: /home/toson/projects/cuda_test)
# 1_cuda_device_parameters
nvcc -o "1_cuda_device_parameters" "1_cuda_device_parameters.cu"
# 2_cuda_2floats_add
nvcc -o "2_cuda_2floats_add" "2_cuda_2floats_add.cu"
# 3_cuda_2matrix_mul
nvcc -o "3_cuda_2matrix_mul" "3_cuda_2matrix_mul.cu"
# 4_cuda_51cto_Cuda编程(并行计算)视频课程_5
nvcc -o "4_cuda_51cto_Cuda编程(并行计算)视频课程_5" "4_cuda_51cto_Cuda编程(并行计算)视频课程_5.cu"
...
```

运行：
```shell
./1_cuda_device_parameters
./2_cuda_2floats_add
./3_cuda_2matrix_mul
./4_cuda_51cto_Cuda编程(并行计算)视频课程_5
...
```

