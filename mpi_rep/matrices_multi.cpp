#include <mpi.h>
#include <memory>
#include <iostream>
#include <cstdlib> // Para rand() y srand()
#include <ctime>   // Para time()
#include <fmt/core.h>

#define MATRIX_DIM 25

void imprimir_vector(const std::string msg, double* v, int size) {
    fmt::print("{} [",msg);
        for (int i=0; i<size;i++) {
            fmt::print("{},",v[i]);
        }

    fmt::println("]");
}


int main(int argc, char* argv[]) {


    MPI_Init(&argc, &argv); // Inicializa MPI
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el rank del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Obtener el número total de procesos


    int rows_per_rank; //Numero de filas a enviar por cada RANK
    int rows_alloc=MATRIX_DIM; //tamanio ajustado de la matriz, para que el # de filas sea divisipa para el NPROCS
    int padding=0; //numero de filas que se agregan para ajustar el tamanio de la matriz

    if(MATRIX_DIM%nprocs!=0) {
        //el numero de filas no es divisiple para NPROCS
        rows_alloc=std::ceil((double) MATRIX_DIM/nprocs)*nprocs;
        padding=rows_alloc-MATRIX_DIM;
    }

    rows_per_rank=rows_alloc/nprocs;

    //--buffers
    // b= A*x
    std::unique_ptr<double[]> A; //solo RANK_0
    std::unique_ptr<double[]> b; //solo RANK_0
    std::unique_ptr<double[]> x=std::make_unique<double[]>(MATRIX_DIM);
        //permite reservar memoria

    //bi = Ai*x, donde Ai e suna matriz de dimencion 7x25 ==> rows_per_rank x MATRIX_DIM
    /*
    A_0 = [
        f0
        f1
        f2
        f3
        f4
        f5
        f6
    ]

    A_1 = [
        f7
        f8
        f9
        f10
        f11
        f12
        f13
    ]

    A_2 = [
        f14
        f15
        f16
        f17
        f18
        f19
        f20
    ]

    A_3 = [
        f21
        f22
        f23
        f24
        f25 --> padding
        f26 --> padding
        f27 --> padding
    ]

     */

    std::unique_ptr<double[]> A_local;
    std::unique_ptr<double[]> b_local;

    if(rank==0) {
        std::printf("Dimension=%d, rows_alloc=%d, rows_per_rank=%d, padding=%d\n",
            MATRIX_DIM, rows_alloc, rows_per_rank, padding
            );
        //fmt::print("Dimension={}, rows_alloc={}, rows_per_rank={}, padding={}\n",
        //    MATRIX_DIM, rows_alloc, rows_per_rank, padding
        //    );

        A=std::make_unique<double[]>(rows_alloc*MATRIX_DIM);
        b=std::make_unique<double[]>(rows_alloc);

        //--incializar matriz A,, vector x
         std::printf("        c1 c2 c3 ...\n");
         for(int i=0; i<MATRIX_DIM; i++) {
             fmt::print("fila{:>3} |",i);
             for(int j=0; j<MATRIX_DIM; j++) {
                 int index = i*MATRIX_DIM + j;
                 A[index]=i;
                 fmt::print( "{:>2} ", A[index]);
             }
             std::printf("| \n");
         }

        for (int i=0; i<MATRIX_DIM; i++) {
            x[i]=1;
        }

        //--incializar matriz A,, vector x
        // std::printf("        c1 c2 c3 ...\n");
        // for(int i=0; i<MATRIX_DIM; i++) {
        //     std::printf("fila %d  |",i);
        //     for(int j=0; j<MATRIX_DIM; j++) {
        //         int index = i*MATRIX_DIM + j;
        //         A[index]=i;
        //         std::printf("%.0f  ", A[index]);
        //     }
        //     std::printf("| \n");
        // }
    }

    //--incializar matrizes locales
    A_local=std::make_unique<double[]>(rows_per_rank*MATRIX_DIM);
    b_local=std::make_unique<double[]>(rows_per_rank);

    //imprimir vector 'x'
    auto txt = fmt::format("RANK_{}",rank);
    imprimir_vector(txt,x.get(),MATRIX_DIM);

    MPI_Bcast(x.get(), MATRIX_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto txt_after = fmt::format("RANK_{} after",rank);
    imprimir_vector(txt_after,x.get(),MATRIX_DIM);


    MPI_Finalize();
    return 0;

}