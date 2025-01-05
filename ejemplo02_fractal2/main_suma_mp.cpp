#include <iostream>
#include <vector>
#include <omp.h>

// Generar un vector de prueba con valores del 1 al N
std::vector<float> generate_vector(int n) {
    std::vector<float> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = i + 1;
    }
    return vec;
}

// Implementación de reducción con OpenMP
float sum_reduction_openmp(std::vector<float>& input) {
    int n = input.size();
    while (n > 1) {
        int num_threads = n / 2;
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_threads; ++i) {
            input[i] += input[i + num_threads];
        }
        n /= 2; // Reducimos el tamaño a la mitad
    }
    return input[0];
}

int main_suma_mp() {
    const int N = 8; // Tamaño del vector
    std::vector<float> input = generate_vector(N);

    // OpenMP
    std::vector<float> input_openmp = input; // Copia del vector para OpenMP
    float result_openmp = sum_reduction_openmp(input_openmp);
    std::cout << "Resultado con OpenMP: " << result_openmp << "\n";

    return 0;
}
