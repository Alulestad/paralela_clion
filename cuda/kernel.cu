
#include <iostream>

const int PALETE_SIZE = 16;

//es para ponerlo en la memoria de la GPU
__constant__
unsigned int d_Pallete[PALETE_SIZE];


__device__
unsigned int divergente(double cx, double cy, int max_iterations) {
    int iter = 0;
    double vx = cx;
    double vy = cy;

    while (iter<max_iterations && (vx * vx + vy * vy) <= 4.0) {
        double tx = vx * vx - vy * vy + cx;
        double ty = 2.0 * vx * vy + cy;
        vx = tx;
        vy = ty;
        iter++;
    }

    if (iter > 0 && iter < max_iterations) {
        int color_idx = iter % PALETE_SIZE;
        return d_Pallete[color_idx];
    }

    if ((vx * vx + vy * vy) > 4.0) {
        return d_Pallete[0]; // Color para puntos que se escapan
    }

    return 0;  // Converge (fuera del conjunto de Mandelbrot)
}

__global__
void mandelbrotKernel(unsigned int* buffer,
    unsigned int width, unsigned int height,
    double x_min, double x_max, double y_min, double y_max,
    double dx, double dy,
    int  max_iterations
     ) {
    // Calculamos el paso en el eje x y en el eje
    // y según el rango definido y las dimensiones de la ventana.
    // dx = (x_max - x_min) / width;
    // dy = (y_max - y_min) / height;

    // #pragma omp parallel for num_threads(16)
    //#pragma omp parallel for collapse(2)
    //Colapsa dos bucles anidados en un único bucle para mejorar
    //la distribución de trabajo entre hilos.
    //#pragma omp parallel for default(none)
    //#pragma omp parallel for  shared(pixel_buffer,dx,dy)

    //#pragma omp parallel for default(none) shared(pixel_buffer,dx,dy)
    // - `default(none)` asegura que todas las variables utilizadas en el bloque deben estar explícitamente declaradas.
    // - `shared(pixel_buffer, dx, dy)` especifica que estas variables son compartidas entre los hilos.


    // Convertimos las coordenadas del píxel (i, j) a coordenadas del plano complejo.

    unsigned id=blockIdx.x*blockDim.x+threadIdx.x;

    if (id<width*height) {
        // int i=id%width;
        // int j=id/width;

        int i=id/width;
        int j=id%width;

        double x = x_min + i * dx;
        double y = y_max - j * dy;  // Invertimos y para que la parte superior sea `y_max`.

        // Calculamos el color de acuerdo a la función de divergencia.
        uint32_t color = divergente(x, y,max_iterations);
        // Asignamos el color al buffer de píxeles en la posición correspondiente.
        buffer[j * width + i] = color;
    }
}


extern "C"
void copy_pallete_to_gpu(unsigned int* h_pallete){
    cudaMemcpyToSymbol(d_Pallete, h_pallete, PALETE_SIZE * sizeof(unsigned int));
}

extern "C"
void mandelbrotGpuKernel(unsigned int* buffer,
    unsigned int width, unsigned int height,
    double x_min, double x_max, double y_min, double y_max,
    double dx, double dy,
    int  max_iterations) {
    // double dx = (x_max - x_min) / width;
    // double dy = (y_max - y_min) / height;

    int threads_per_block = 1024;
    int blocks_in_grid = std::ceil(float(width * height) / threads_per_block);

    printf("blocks=%d, threads=%d\n",blocks_in_grid,threads_per_block);

    mandelbrotKernel<<<blocks_in_grid,threads_per_block>>>(buffer,
        width, height,
        x_min, x_max, y_min, y_max,
        dx, dy,
        max_iterations);

}
