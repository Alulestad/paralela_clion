#include <mpi.h>
#include <iostream>
#include <cstdlib> // Para rand() y srand()
#include <ctime>   // Para time()

int random_value() {
    srand(time(nullptr)); // Inicializar semilla para rand()
    return rand() % 100; // Generar un número aleatorio entre 0 y 99
}

int main(int argc, char* argv[]) {
    int rank, size, value;

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el rank del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtener el número total de procesos

    std::cout<<"Rank: "<<rank<<" Size: "<<size<<std::endl;

    if (size < 2) {
        std::cerr << "Este programa requiere al menos 2 procesos." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        // Proceso 0 genera un valor inicial aleatorio
        value = random_value();
        std::cout << "Proceso " << rank << " genero' valor inicial: " << value << std::endl;

        // Enviar el valor al siguiente proceso
        MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }else{
        // Resto de los procesos
        MPI_Recv(&value, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value++; // Incrementar el valor
        std::cout << "Proceso " << rank << " recibio' valor: " << value << std::endl;

        // Enviar el valor al siguiente proceso
        MPI_Send(&value, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    }
    // Proceso 0 recibe el valor final del último proceso
    if (rank == 0) {
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proceso " << rank << " recibio' valor final: " << value << std::endl;
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}