
#include<iostream>

__device__
int aux(int x) {
    return x*2;
}

//permite desde el host invocar a la gpu
// no tendris sentid __device__ por el momento pues solo se invocaria
//dentor del device
__global__
void suma_kernel(float* a, float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    aux(1);

    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

extern "C"
void suma_paralela(float* a, float* b, float* c, int size) {
    int thread_num=1024;
    int block_sum=std::ceil(size/ (float)thread_num);

    std::printf("num_bloks: %d, num_threads: %d\n", block_sum, thread_num);

    suma_kernel<<<block_sum, 256>>>(a, b, c, size);

}
