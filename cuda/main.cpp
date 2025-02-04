#include <iostream>

#include <cuda_runtime.h>
#include <iostream>
#include <fmt/core.h>
#include <sfml/graphics.hpp>
#include <cuda_runtime.h>



//--------------------------------------------------------
// Función para invertir los bytes de un entero de 32 bits
uint32_t _bswap32(uint32_t a) {
    return
    ((a & 0X000000FF) << 24) |  // Mueve el byte menos significativo al más significativo
        ((a & 0X0000FF00) <<  8) |  // Mueve el segundo byte a la posición correcta
            ((a & 0x00FF0000) >>  8) |  // Mueve el tercer byte hacia la derecha
                ((a & 0xFF000000) >> 24);  // Mueve el byte más significativo al menos significativo
}

//--------------------------------------------------------
// Definición de constantes
#define WIDTH 1600  // Ancho de la ventana
#define HEIGHT 900  // Altura de la ventana

// Límites del plano cartesiano para calcular Mandelbrot
const double x_max = 1;
const double x_min = -2;
const double y_min = -1;
const double y_max = 1;

const int max_iterations = 100;  // Iteraciones máximas para determinar convergencia
const int PALETE_SIZE = 16;  // Tamaño de la paleta de colores

// Paleta de colores en formato RGBA almacenados como enteros de 32 bits
const uint32_t color_ramp[PALETE_SIZE] = {
    _bswap32(0xFFFF1010),
    _bswap32(0xFFF31017),
    _bswap32(0xFFE8101E),
    _bswap32(0xFFDC1126),
    _bswap32(0xFFD1112D),
    _bswap32(0xFFC51235),
    _bswap32(0xFFBA123C),
    _bswap32(0xFFAE1343),
    _bswap32(0xFFA3134B),
    _bswap32(0xFF971452),
    _bswap32(0xFF8C145A),
    _bswap32(0xFF801461),
    _bswap32(0xFF751568),
    _bswap32(0xFF691570),
    _bswap32(0xFF5E1677),
    _bswap32(0xFF54167D),
};

static unsigned int* host_pixel_buffer=nullptr;
static unsigned int* device_pixel_buffer=nullptr;

//--------------------------------------------------------



//--------------------------------------------------------
//mandelbrot
int divergente(double cx, double cy) {
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
        return color_ramp[color_idx];
    }

    if ((vx * vx + vy * vy) > 4.0) {
        return color_ramp[0]; // Color para puntos que se escapan
    }

    return 0;  // Converge (fuera del conjunto de Mandelbrot)
}

void mandelbrotCpu() {
    // Calculamos el paso en el eje x y en el eje
    // y según el rango definido y las dimensiones de la ventana.
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

// #pragma omp parallel for num_threads(16)
//#pragma omp parallel for collapse(2)
    //Colapsa dos bucles anidados en un único bucle para mejorar
    //la distribución de trabajo entre hilos.
//#pragma omp parallel for default(none)
//#pragma omp parallel for  shared(pixel_buffer,dx,dy)

#pragma omp parallel for default(none) shared(pixel_buffer,dx,dy)
// - `default(none)` asegura que todas las variables utilizadas en el bloque deben estar explícitamente declaradas.
// - `shared(pixel_buffer, dx, dy)` especifica que estas variables son compartidas entre los hilos.
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            // Convertimos las coordenadas del píxel (i, j) a coordenadas del plano complejo.
            double x = x_min + i * dx;
            double y = y_max - j * dy;  // Invertimos y para que la parte superior sea `y_max`.

            // Calculamos el color de acuerdo a la función de divergencia.
            int color = divergente(x, y);
            // Asignamos el color al buffer de píxeles en la posición correspondiente.
            //pixel_buffer[j * WIDTH + i] = color;
        }
    }

}

//--------------------------------------------------------

#define CHECK(expr){           \
    auto internal_error = (expr);        \
    if (internal_error!=cudaSuccess){    \
        fmt::println("{}: {} in (en) {} at line (en la linea) {}",(int )error, cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);     \
    }                           \
}                               \


int main() {

    int device=0;
    cudaSetDevice(device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props,device);

    fmt::println("Device GB {}",props.name);
    fmt::println("Total memory {}GB",props.totalGlobalMem/1024/1024/1024);
    fmt::println("Multiprocessors {}",props.multiProcessorCount);
    fmt::println("Max threads for multiprocessor {}",props.maxThreadsPerMultiProcessor);
    fmt::println("Max threads per block {}",props.maxThreadsPerBlock);

    //--inicializar
    size_t buffer_size=WIDTH*HEIGHT*sizeof(unsigned int); // el tamanio del buffer

    host_pixel_buffer=(unsigned int *) malloc(buffer_size); // reservar memoria en el host es decir la ram
    cudaError_t error=cudaMalloc(&device_pixel_buffer,buffer_size); // reservar memoria en el device es decir la targeta grafica
    CHECK(error); //si lo hacemos como macro
    // esta copia el codigo de la macro a esta linea

    //check(error); //si lo hacemos como metodo
    //siempre me dara una misma linea, pues se ejecuta en el metodo



    // Create the main window
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Mandelrrot CUDA");


    // Start the game loop
    while (window.isOpen())
    {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Clear screen
        window.clear();

        // Update the window
        window.display();
    }


    return 0;
}


