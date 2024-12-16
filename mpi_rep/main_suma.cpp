#include <iostream>
#include <vector>
#include <mpi.h>

// Generar un vector de prueba con valores del 1 al N
std::vector<float> generate_vector(int n) {
    std::vector<float> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = i + 1;
    }
    return vec;
}

// Implementación de reducción con MPI
float sum_reduction_mpi(std::vector<float>& input, int rank, int size) {
    int n = input.size();
    int elements_per_proc = n / size;

    // Dividir el trabajo entre procesos
    std::vector<float> local_data(elements_per_proc);
    MPI_Scatter(input.data(), elements_per_proc, MPI_FLOAT, local_data.data(), elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Reducción local
    float local_sum = 0.0f;
    for (float value : local_data) {
        local_sum += value;
    }

    // Reducción global
    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_sum;
}

int main_suma(int argc, char** argv) {
    const int N = 8; // Tamaño del vector
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> input;
    if (rank == 0) {
        input = generate_vector(N);
    }

    float result_mpi = sum_reduction_mpi(input, rank, size);
    if (rank == 0) {
        std::cout << "Resultado con MPI: " << result_mpi << "\n";
    }

    MPI_Finalize();

    return 0;
}
