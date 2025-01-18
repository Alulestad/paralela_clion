#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm> // Para std::min y std::max
#include <numeric>   // Para std::accumulate
#include <fmt/core.h>
#include <omp.h>     // Para OpenMP

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

// Función para calcular y mostrar la tabla de frecuencias (serial)
void calculate_frequencies_serial(const std::vector<int>& data) {
    std::vector<int> frequencies(101, 0); //uso el constructor con 101 elementos 0, 1, 2, .... 100 y valor 0

    // Contar frecuencias de forma serial
    for (int value : data) {
        frequencies[value]++;
    }

    // Imprimir tabla de frecuencias
    fmt::print("+-------+--------+\n");
    fmt::print("| Valor | Conteo |\n");
    fmt::print("+-------+--------+\n");
    for (size_t i = 0; i < frequencies.size(); ++i) {
        fmt::print("| {:5d} | {:6d} |\n", i, frequencies[i]);
    }
    fmt::print("+-------+--------+\n");
}

// Función para calcular el promedio, mínimo y máximo (serial)
void calculate_statistics_serial(const std::vector<int>& data) {
    // Calcular promedio
    double average = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    // Calcular mínimo y máximo
    int min_value = *std::min_element(data.begin(), data.end());
    int max_value = *std::max_element(data.begin(), data.end());

    fmt::print("Promedio (serial): {:.2f}\n", average);
    fmt::print("Mínimo (serial): {}\n", min_value);
    fmt::print("Máximo (serial): {}\n", max_value);
}

// Función para calcular y mostrar estadísticas usando OpenMP secciones
void calculate_statistics_parallel_sections(const std::vector<int>& data) {
    double average = 0.0;
    int min_value = 0, max_value = 0;

    #pragma omp parallel sections
    {
        // Calcular promedio en una sección
        #pragma omp section
        {
            average = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        }

        // Calcular mínimo en otra sección
        #pragma omp section
        {
            min_value = *std::min_element(data.begin(), data.end());
        }

        // Calcular máximo en otra sección
        #pragma omp section
        {
            max_value = *std::max_element(data.begin(), data.end());
        }
    }

    fmt::print("Promedio (paralelo - secciones): {:.2f}\n", average);
    fmt::print("Mínimo (paralelo - secciones): {}\n", min_value);
    fmt::print("Máximo (paralelo - secciones): {}\n", max_value);
}


// Función para calcular frecuencias usando OpenMP bucles paralelos
void calculate_frequencies_parallel(const std::vector<int>& data) {
    std::vector<int> frequencies(101, 0);

    #pragma omp parallel
    {
        // Crear un vector de frecuencias local para cada hilo
        std::vector<int> local_frequencies(101, 0);

        #pragma omp for
        for (int i = 0; i < static_cast<int>( data.size()); ++i) {
            //OpenMP requiere un índice de bucle con signo porque su especificación está diseñada para trabajar con tipos enteros firmados.

            local_frequencies[data[i]]++;
        }

        // Reducir los resultados locales a un vector global
        #pragma omp critical
        {
            for (size_t i = 0; i < frequencies.size(); ++i) {
                frequencies[i] += local_frequencies[i];
            }
        }
    }

    // Imprimir tabla de frecuencias
    fmt::print("+-------+--------+\n");
    fmt::print("| Valor | Conteo |\n");
    fmt::print("+-------+--------+\n");
    for (size_t i = 0; i < frequencies.size(); ++i) {
        fmt::print("| {:5d} | {:6d} |\n", i, frequencies[i]);
    }
    fmt::print("+-------+--------+\n");
}

int main() {
    // Leer los datos del archivo
    const std::string file_path = "C:/Users/Dami/Documents/CLionProjects/mpi_rep/datos.txt";
    std::vector<int> data = read_file(file_path);

    // Calcular frecuencias y estadísticas de forma serial
    calculate_frequencies_serial(data);
    calculate_statistics_serial(data);

    // Calcular estadísticas con OpenMP (secciones paralelas)
    calculate_statistics_parallel_sections(data);

    // Calcular frecuencias con OpenMP (bucles paralelos)
    calculate_frequencies_parallel(data);

    return 0;
}
