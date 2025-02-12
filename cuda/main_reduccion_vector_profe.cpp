
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <fmt/core.h>

#define MAX_ELEMENTOS 1024

#define CHECK(expr){                    \
auto error = expr;                  \
if (error != cudaSuccess) {         \
fmt::print("{}: {} in {} at line {}\n", (int)error, (char*)cudaGetErrorString(error), __FILE__, __LINE__); \
exit(EXIT_FAILURE);             \
}                                   \
}

extern "C"
void gpu_reducction(int* input, int* output,int block_per_grid,int threads_per_block, int size);

int main(){
    std::vector<int> input(MAX_ELEMENTOS);
    std::generate(input.begin(), input.end(), [n = 0]() mutable { return n++; });

    //--
    int threads_per_block=1024; //256
    int blocks_per_grid=MAX_ELEMENTOS/threads_per_block;

    size_t size_in_bytes=MAX_ELEMENTOS*sizeof(int);

    int* d_input; // 10240 elementos
    int* d_output;// 10 elementos
    int* d_suma;  // 1 elemento

    CHECK(cudaMalloc((void**)&d_input, size_in_bytes));
    CHECK(cudaMalloc((void**)&d_output, blocks_per_grid*sizeof(int)));
    CHECK(cudaMalloc((void**)&d_suma, 1*sizeof(int)));


    //--copiar los datos
    CHECK(cudaMemcpy(d_input, input.data(), size_in_bytes, cudaMemcpyHostToDevice));


    //--invocar kernel
    gpu_reducction(d_input, d_output, blocks_per_grid, threads_per_block, MAX_ELEMENTOS);
    CHECK(cudaGetLastError());
    gpu_reducction(d_output,d_suma,1,blocks_per_grid,blocks_per_grid);
    CHECK(cudaGetLastError());

    //--copiar los datos d evuelta
    std::vector<int> tmp(blocks_per_grid);
    int suma_final;

    CHECK(cudaMemcpy(tmp.data(),d_output, blocks_per_grid*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&suma_final,d_suma,1*sizeof(int), cudaMemcpyDeviceToHost));

    fmt::print("{}", tmp);
    // fmt::println("suma: {}",suma_final);
    ///std::cout << "tmp: " << tmp << std::endl;
    std::cout << "suma: " << suma_final << std::endl;


    return 0;

    //HACER como ejercicio
    // generar un histograma en la OMP
    // y la GPU

}


