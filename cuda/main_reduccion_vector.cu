#include <algorithm>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define MAX_ELEMENTOS 8

// Kernel para la reducción paralela
__global__ void sum_reduction(int* data, int n) {
    int tid = threadIdx.x;

    // Realizar la reducción en paralelo
    for (int stride = n / 2; stride > 0; stride /= 2) {
        __syncthreads(); // Sincronizar los hilos antes de acceder a la memoria compartida

        if (tid < stride) {
            data[tid] += data[tid + stride];
        }
    }
}

int main() {
    // Inicializar el vector de entrada
    std::vector<int> input(MAX_ELEMENTOS);
    std::generate(input.begin(), input.end(), [n = 0]() mutable { return n++; });

    // Imprimir el vector inicial
    std::cout << "Vector de entrada: ";
    for (const auto& val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Copiar datos al device (GPU)
    int* d_data;
    size_t size = MAX_ELEMENTOS * sizeof(int);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);

    // Lanzar el kernel con tantos hilos como elementos
    sum_reduction<<<1, MAX_ELEMENTOS>>>(d_data, MAX_ELEMENTOS);

    // Copiar el resultado de vuelta al host (CPU)
    int result;
    cudaMemcpy(&result, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria en el device
    cudaFree(d_data);

    // Imprimir el resultado
    std::cout << "Suma: " << result << std::endl;

    return 0;
}
