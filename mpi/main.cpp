#include <iostream>
#include <mpi.h>
int main( int argc, char** argv ){
    MPI_Init(&argc, &argv);

    int rank;
    int nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //ID del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); //# de procesos

    //std::printf("Rank: %d, nprocs: %d\n", rank, nprocs);
    //while (true){}
    int valor= 0;
    int temp=0;
    if (rank == 0){
        valor = 99;
        //MPI_Send(&valor, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        //MPI_Send(&valor, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        //MPI_Send(&valor, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);

        for(int rankID=1;  rankID<nprocs; rankID++){
            MPI_Send(&valor, 1, MPI_INT, rankID, 0, MPI_COMM_WORLD);
        }
    } else{

        MPI_Recv(&temp, 1 , MPI_INT,0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    }
    //std::printf("RANK: %d, valor: %d\n", rank, valor);
    std::printf("RANK_%d, recibido el valor: %d\n", rank, temp);
    MPI_Finalize();
    return 0;
}
