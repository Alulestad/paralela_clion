#include <iostream>

#include <cuda_runtime.h>
#include <iostream>
#include <fmt/core.h>
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
//#include "fps_counter.h"




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

static unsigned int* host_pixel_buffer = nullptr;
static unsigned int* device_pixel_buffer = nullptr;

//--------------------------------------------------------



#define CHECK(expr){           \
    auto error = (expr);        \
    if (error!=cudaSuccess){    \
        fmt::println("{}: {} in (en) {} at line (en la linea) {}",(int )error, (char*)cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);     \
    }                           \
}                               \


//mandelbrot
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

//#pragma omp parallel for default(none) shared(pixel_buffer,dx,dy)
// - `default(none)` asegura que todas las variables utilizadas en el bloque deben estar explícitamente declaradas.
// - `shared(pixel_buffer, dx, dy)` especifica que estas variables son compartidas entre los hilos.
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            // Convertimos las coordenadas del píxel (i, j) a coordenadas del plano complejo.
            double x = x_min + i * dx;
            double y = y_max - j * dy;  // Invertimos y para que la parte superior sea `y_max`.

            // Calculamos el color de acuerdo a la función de divergencia.
            unsigned int color = divergente(x, y,max_iterations);
            // Asignamos el color al buffer de píxeles en la posición correspondiente.
            host_pixel_buffer[j * WIDTH + i] = color;
        }
    }

}


extern "C" void mandelbrotGpuKernel(unsigned int* buffer,
    unsigned int width, unsigned int height,
    double x_min, double x_max, double y_min, double y_max,
    double dx, double dy,
    int max_iterations);


extern "C" void copy_pallete_to_gpu(unsigned int* h_pallete);



void mandelbrotGpu() {
    // Calcular dx y dy fuera del kernel
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;
    // Llamar al kernel con todos los parámetros
    mandelbrotGpuKernel(device_pixel_buffer, WIDTH, HEIGHT, x_min, x_max, y_min, y_max, dx, dy, max_iterations);
    // Verificar errores
    CHECK(cudaGetLastError());
    // Copiar el resultado desde la GPU al host

    CHECK(cudaMemcpy(host_pixel_buffer, device_pixel_buffer, WIDTH * HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost));

}






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
    memset(host_pixel_buffer,0,buffer_size);// inzializamos con ceros

    //cudaError_t error=cudaMalloc(&device_pixel_buffer,buffer_size); // reservar memoria en el device es decir la targeta grafica
    CHECK(cudaMalloc(&device_pixel_buffer,buffer_size)); //si lo hacemos como macro
    // esta copia el codigo de la macro a esta linea

    //check(error); //si lo hacemos como metodo
    //siempre me dara una misma linea, pues se ejecuta en el metodo



    // Create the main window
    // -- inicializacion de la interfaz grafia
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML Mandelrrot CUDA");


    copy_pallete_to_gpu((unsigned int*)color_ramp);
    mandelbrotGpu();


    //creamos la textura
    sf::Texture texture;
    texture.create(WIDTH,HEIGHT);
    texture.update((const sf::Uint8 *) host_pixel_buffer);

    sf::Sprite sprite;
    sprite.setTexture(texture);


    sf::Font font;
    font.loadFromFile("../arial.ttf"); //cargamos la fuente
    sf::Text text;
    {

        text.setFont(font); //fuente del texto
        text.setString("Mandelbrot set");
        text.setCharacterSize(24); //tamanio del texto
        text.setFillColor(sf::Color::White); //color del texto
        text.setStyle(sf::Text::Bold); //estilo del texto
        text.setPosition(10,10); //posicion del texto
    }

    // --FPS
    sf::Clock clock;
    int frames=0;
    int fps=0;

    // Start the game loop
    while (window.isOpen()) {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed)
                window.close();
        }

        //--regenrar el dibujo
        mandelbrotCpu();
        texture.update((const sf::Uint8 *) host_pixel_buffer);

        // --contador FPS
        frames++;

        if (clock.getElapsedTime().asSeconds()>=1) {
            fps=frames;
            frames=0;
            clock.restart();
        }

        auto msg = fmt::format("Mandelbrot set - Iterations={} - FPS: {}",max_iterations,fps);
        text.setString(msg);
        // Clear screen
        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
        }

        // Update the window
        window.display();
    }


    return 0;
}


