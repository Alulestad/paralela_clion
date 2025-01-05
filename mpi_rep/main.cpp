#include <mpi.h>
#include <iostream>
#include <cstdlib> // Para rand() y srand()
#include <ctime>   // Para time()


#define MATRIX_DIM 25


int random_value() {
    srand(time(nullptr)); // Inicializar semilla para rand()
    return rand() % 100; // Generar un número aleatorio entre 0 y 99
}
int main(int argc, char* argv[]) {
    int rank, size, value;


    return 0;
}