__global__
void kernel_reduccion(int* input, int* output,int size) {

    int index=threadIdx.x+blockDim.x*blockIdx.x;

    for (int it=blockDim.x/2;it>0;it>>=1){

        if (threadIdx.x<it){
            input[index]= input[index] + input[index+it];
        }

        __syncthreads();
    }


    if (threadIdx.x==0)
        output[blockIdx.x]=input[index];
}

extern "C"
void gpu_reducction(int* input, int* output,int block_per_grid,int threads_per_block, int size) {
    kernel_reduccion<<<block_per_grid,threads_per_block>>>(input,output,size);
}