#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h> // Para utilizar MPI
#include <numeric>   // Para std::accumulate
#include <algorithm> // Para std::min y std::max
#include <iomanip>



// Función para leer los datos desde un archivo
std::vector<int> read_file(const std::string& file_path) {
    std::fstream fs(file_path, std::ios::in);
    std::string line;

    std::vector<int> data;
    while (std::getline(fs, line)) {
        data.push_back(std::stoi(line));
    }
    fs.close();

    return data;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Inicializa MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Identificador del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Número total de procesos

    const std::string file_path = "C:/Users/Dami/Documents/CLionProjects/mpi_rep/datos.txt";
    std::vector<int> data;
    std::vector<int> local_data;

    // Solo el proceso 0 lee los datos
    if (rank == 0) {
        data = read_file(file_path);
    }

    // Scatter: distribuir datos a los procesos
    int total_elements = (rank == 0) ? data.size() : 0;
    int elements_per_proc = total_elements / size;
    local_data.resize(elements_per_proc);
    MPI_Scatter(data.data(), elements_per_proc, MPI_INT, local_data.data(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // 1. Calcular tabla de frecuencias localmente
    std::vector<int> local_frequencies(101, 0);
    for (int value : local_data) {
        local_frequencies[value]++;
    }

    // Reducir las frecuencias globales en el proceso 0
    std::vector<int> global_frequencies(101, 0);
    MPI_Reduce(local_frequencies.data(), global_frequencies.data(), 101, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Imprimir tabla de frecuencias
        std::cout << "+-------+--------+\n";
        std::cout << "| Valor | Conteo |\n";
        std::cout << "+-------+--------+\n";
        for (size_t i = 0; i < global_frequencies.size(); ++i) {
            std::cout << "| " << std::setw(5) << i << " | " << std::setw(6) << global_frequencies[i] << " |\n";
        }
        std::cout << "+-------+--------+\n";
    }

    // 2. Calcular promedio local
    double local_sum = std::accumulate(local_data.begin(), local_data.end(), 0.0);
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double average = global_sum / total_elements;
        std::cout << "Promedio: " << average << "\n";
    }

    // 3. Calcular mínimo y máximo localmente
    int local_min = *std::min_element(local_data.begin(), local_data.end());
    int local_max = *std::max_element(local_data.begin(), local_data.end());

    int global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Minimo: " << global_min << "\n";
        std::cout << "Maximo: " << global_max << "\n";
    }

    MPI_Finalize(); // Finaliza MPI
    return 0;
}
