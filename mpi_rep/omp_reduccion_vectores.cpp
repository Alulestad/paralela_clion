
#include <iostream>
#include <vector>



int main(){
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    int numThreads = vec.size()/2;

// #pragma omp parallel for reduction(+:vec)
    #pragma omp parallel for num_threads(numThreads)
    for (int i=0; i<vec.size();i++) {
        std::cout<<"Impresion para ver: "<<vec[i]<<std::endl;

    }


    return 0;
}


float sum_reduction(const float *input, const int n) {
    float suma = 0.0f;
    for (int i = 0; i < n; ++i)
        suma = suma + input[i];
    return suma;
}