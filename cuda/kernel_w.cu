#include <iostream>
const int PALETTE_SIZE = 16;

//
__constant__ //bloque de memoria en la GPU
unsigned int d_Pallete[PALETTE_SIZE];

__device__
unsigned int divergente(double cx, double cy, int max_iterations) {

    int iter = 0;

    double vx = cx;
    double vy = cy;

    while(iter<max_iterations && (vx*vx+vy*vy)<=4) {
        //Zn+1 = Zn^2 + C
        double tx = vx * vx - vy * vy + cx; //vx^2-vy^2+cx
        double ty = 2 * vx * vy + cy; // 2 vx vy + cy

        vx = tx;
        vy = ty;

        iter++;
    }

    if(iter>0 && iter<max_iterations) {
        // diverge
        int color_idx = iter % PALETTE_SIZE;
        return d_Pallete[color_idx];
    }

    if((vx*vx+vy*vy)>4) {
        return d_Pallete[0];
    }

    // convergente
    return 0;
}

extern "C"
void copy_pallete(unsigned int* h_pallete)
{
    cudaMemcpyToSymbol(d_Pallete, h_pallete, PALETTE_SIZE*sizeof(unsigned int));
}

__global__
void mandelbrotKernel(unsigned int* buffer,
unsigned int width, unsigned height
, double x_min, double x_max, double y_min,
double y_max, double dx, double dy, int max_iterations)
{
    // double dx = (x_max-x_min) / WIDTH;
    // double dy = (y_max-y_min) / HEIGHT;

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < width*height)
    {
        int j = id/width;
        int i = id%width;

        double x = x_min + i*dx;
        double y = y_max - j*dy;

        // C = X+Yi
        unsigned int color = divergente(x,y, max_iterations);
        buffer[j*width + i] = color;
    }
}

extern "C"
void mandelbrotGPUkernel(unsigned int* buffer,
unsigned int width, unsigned height
, double x_min, double x_max, double y_min,
double y_max, int max_iterations)
{
    double dx = (x_max-x_min) / width;
    double dy = (y_max-y_min) / height;

    int threds_per_block = 1024;
    int blocks_in_grid = std::ceil(float(width*height/threds_per_block));

    mandelbrotKernel<<<blocks_in_grid, threds_per_block>>>(buffer,
        width, height,
        x_min, x_max, y_min, y_max,
        dx, dy,
        max_iterations);


}