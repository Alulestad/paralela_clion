#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>
#include <fmt/core.h>  // Corrección aquí: se debe incluir <fmt/core.h> en lugar de <fmt/base.h>

#define MAX_ELEMENTOS 8

int main() {
    std::vector<int> input(MAX_ELEMENTOS);

    std::generate(input.begin(), input.end(), [n = 0]() mutable { return n++; });

    int num_threads = MAX_ELEMENTOS / 2;

    // Copiar el vector a un temporal
    std::vector<int> tmp_vector;
    std::copy(input.begin(), input.end(), std::back_inserter(tmp_vector));

    //fmt::print("{}\n", tmp_vector);  // Corrección aquí: fmt::println no existe, se utiliza fmt::print con \n

    int* data = tmp_vector.data();
    int suma = 0;  // Corrección aquí: inicializar suma

#pragma omp parallel shared(data, suma,tmp_vector) default(none) num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        for (int it = num_threads; it > 0; it /= 2) {

            // #pragma omp master
            // fmt::print(fg(fmt::color::blue), "iteracion {}\n", it);

            if (thread_id < it) {
                data[thread_id] = data[thread_id] + data[thread_id + it];
            }
#pragma omp barrier

            // #pragma omp master
            //     fmt::println("{}",tmp_vector);

        }



        if(thread_id == 0)
            suma=data[0];
        fmt::print("Thread {}: {} + {}\n", thread_id, thread_id, thread_id + num_threads);
    }

    //fmt::print("{}\n", tmp_vector);
    fmt::print("Suma: {}\n", suma);

    return 0;  // Corrección aquí: agregar return 0 al final de main()
}

float sum_reduction(const float* input, const int n) {
    float suma = 0.0f;
    for (int i = 0; i < n; ++i)
        suma = suma + input[i];
    return suma;
}
