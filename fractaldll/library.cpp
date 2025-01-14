#include "library.h"

#include <iostream>

#define WIDTH 1600
#define HEIGHT 900

const double x_min = -2;
const double x_max = 1;
const double y_min = -1;
const double y_max = 1;

const int PALETTE_SIZE = 16;

unsigned int _bswap32(unsigned int a) {
    return
        ((a & 0X000000FF) << 24) |
        ((a & 0X0000FF00) <<  8) |
        ((a & 0x00FF0000) >>  8) |
        ((a & 0xFF000000) >> 24);
}

const unsigned int color_ramp[] = {
    _bswap32(0xFF1010FF),
    _bswap32(0xEF1019FF),
    _bswap32(0xE01123FF),
    _bswap32(0xD1112DFF),
    _bswap32(0xC11237FF),
    _bswap32(0xB21341FF),
    _bswap32(0xA3134BFF),
    _bswap32(0x931455FF),
    _bswap32(0x84145EFF),
    _bswap32(0x751568FF),
    _bswap32(0x651672FF),
    _bswap32(0x56167CFF),
    _bswap32(0x471786FF),
    _bswap32(0x371790FF),
    _bswap32(0x28189AFF),
    _bswap32(0x1919A4FF)
};

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
        return color_ramp[color_idx];
    }

    if((vx*vx+vy*vy)>4) {
        return color_ramp[0];
    }

    // convergente
    return 0;
}

//ejem de convenios:
//__cdecl
//__fastcall
extern "C" __stdcall void mandelbrotCpu(unsigned int* pixel_buffer, int max_iterations) {

#ifdef _DEBUG
    std::printf("mandelbrot C++, max_iterations=%d \n",max_iterations);
    std::std::cout.flush();
#endif

    double dx = (x_max-x_min) / WIDTH;
    double dy = (y_max-y_min) / HEIGHT;

#pragma omp parallel for default(none) shared(pixel_buffer, dx, dy, color_ramp,max_iterations)
    for(int i=0; i<WIDTH; i++) {
        for(int j=0;j<HEIGHT;j++) {
            double x = x_min + i*dx;
            double y = y_max - j*dy;

            // C = X+Yi
            unsigned int color = divergente(x,y, max_iterations);
            pixel_buffer[j*WIDTH + i] = color;
        }
    }
}
