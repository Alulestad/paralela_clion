#include <iostream>

#include <cuda_runtime.h>



//#define VECTOR_SIZE 1024*1024
const size_t VECTOR_SIZE = 1024*1024*2;

void suma_serial(float* x, float* y, float* z, int size) {
    for (int i=0;i<size;i++) {
        z[i]=x[i]+y[i];
    }
}

extern "C"
void suma_paralela(float* x, float* y, float* z, int size);


int main()
{
    //(1,1,1,...,1) + (2,2,2,...,2)=(3,3,3,...,3)
    //--host
    float* h_A= new float[VECTOR_SIZE];
    float* h_B= new float[VECTOR_SIZE];
    float* h_C= new float[VECTOR_SIZE];

    memset(h_C,0,VECTOR_SIZE*sizeof(float));

    for(int i=0; i<VECTOR_SIZE;i++) {
        h_A[i]=1.0f;
        h_B[i]=2.0f;
    }


    //--device
    float* d_A;
    float* d_B;
    float* d_C;

    size_t size_in_bytes=VECTOR_SIZE*sizeof(float);
    //esto me ba a dar el tamanio en bytes
    //size_t es un entero de 64 bits

    //std::printf("%zd \n",size_in_bytes);


    //--alocar memoria en el device
    cudaMalloc(&d_A,size_in_bytes);
    cudaMalloc(&d_B,size_in_bytes);
    cudaMalloc(&d_C,size_in_bytes);

    //-- copiar: host-to.device
    cudaMemcpy(d_A,h_A,size_in_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size_in_bytes,cudaMemcpyHostToDevice);


    //invocar kernel
    suma_paralela(d_A,d_B,d_C,VECTOR_SIZE);


    //--copiar: device-to-host
    cudaMemcpy(h_C,d_C,size_in_bytes,cudaMemcpyDeviceToHost);


    //--imprimir resultado
    std::printf("resultad: ");
    for(int i=0;i<10;i++) {
        std::printf("%.0f ",h_C[i]);
    }

    return 0;
}
