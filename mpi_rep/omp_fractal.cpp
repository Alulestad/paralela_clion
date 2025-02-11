#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>

#include "omp_fps_counter.h"

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
const GLuint color_ramp[PALETE_SIZE] = {
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

//--------------------------------------------------------

// Variables globales para GLFW y OpenGL
static GLFWwindow* window = NULL;  // Puntero a la ventana
GLuint textureID;  // ID de textura de OpenGL
GLuint* pixel_buffer = nullptr;  // Buffer para almacenar colores de los píxeles
omp_fps_counter fps;  // Instancia del contador de FPS


//--------------------------------------------------------
// Función para inicializar las texturas de OpenGL
void initTextures() {
    //tesetura es una imagen 2D o 3D que se mapea sobre una superficie para dar detalle y apariencia.
    // pueden ser colores creo.
    // sí, da colores, sompras, patrones.
    glGenTextures(1, &textureID);  // Genera un identificador para la textura
    glBindTexture(GL_TEXTURE_2D, textureID);  // Enlaza la textura como 2D

    // Define una textura vacía con formato RGBA8 y dimensiones WIDTH x HEIGHT
    glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_RGBA8,
            WIDTH, HEIGHT, 0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            NULL
    );

    // Configura los filtros de la textura
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // GL_LINEAR: Proporciona una mejor calidad visual, especialmente útil para fractales con detalles finos.
    // GL_TEXTURE_MIN_FILTER: Para reducir la textura. Se usa cuando la textura se reduce
    // GL_TEXTURE_MAG_FILTER: Para ampliar la textura. Se usa cuando la textura se amplía



    glBindTexture(GL_TEXTURE_2D, 0);  // Desenlaza la textura
    //Esto evita que operaciones posteriores afecten accidentalmente a esta textura.
}

//--------------------------------------------------------
// Función para inicializar GLFW y configurar la ventana
void init() {
    if (!glfwInit()) { // Inicializa GLFW
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        exit(-1);
    }

    // Crea una ventana con las dimensiones especificadas
    window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGl C++", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window!" << std::endl;
        exit(-1);
    }


    // Configura un callback para manejar eventos de teclado
    glfwSetKeyCallback(window, [](GLFWwindow* window, auto key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);  // Cierra la ventana al presionar ESC
    });

    // Callback para manejar cambios de tamaño de la ventana
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);  // Ajusta la vista al nuevo tamaño
    });


    glfwMakeContextCurrent(window);  // asociar el contexto de OpenGL a una ventana específica.
    //contexto es un entorno que almacena toda la información necesaria para renderizar gráficos
    // en OpenGL, incluidos los estados de renderizado, los shaders y los buffers de geometría.
    std::string version = (char *)glGetString(GL_VERSION);  // Obtiene la versión de OpenGL
    std::string vendor = (char *)glGetString(GL_VENDOR);  // Obtiene el proveedor de GPU
    std::string render = (char *)glGetString(GL_RENDERER);  // Obtiene el nombre del renderizador

    // Imprime información de OpenGL
    fmt::print("OpenGL version supported {}\n", version);
    fmt::print("Vendor : {}\n", vendor);
    fmt::print("Renderer : {}\n", render);

    // cambia al modo de matriz de proyección
    // define cómo los objetos 3D se proyectan en la pantalla 2D.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-1,1);
//cambia el área visible de la escena

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // establece la matriz actual como la matriz identidad, lo que elimina cualquier transformación previa.

    glEnable(GL_TEXTURE_2D); // Habilita texturas 2D

    glfwSwapInterval(1); // Habilita v-sync (sincronización vertical)

    initTextures(); // Inicializa las texturas
}

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
    // Sin default(none), OpenMP asume que todas las variables son compartidas (shared)


    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            // Convertimos las coordenadas del píxel (i, j) a coordenadas del plano complejo.
            double x = x_min + i * dx;
            double y = y_max - j * dy;  // Invertimos y para que la parte superior sea `y_max`.

            // Calculamos el color de acuerdo a la función de divergencia.
            int color = divergente(x, y);
            // Asignamos el color al buffer de píxeles en la posición correspondiente.
            pixel_buffer[j * WIDTH + i] = color;
        }
    }

}

//--------------------------------------------------------

void paint() {
    /*glBegin(GL_TRIANGLES);
    {
        glVertex2d(-1,-1);
        glVertex2d(0,0);
        glVertex2d(0,-1);
    }
    glEnd();*/

    // Actualizamos el contador de fotogramas por segundo.
    fps.update();

    // Calculamos el conjunto de Mandelbrot y almacenamos los datos en el buffer de píxeles.
    mandelbrotCpu();

    /////////////////////////////////////////////// DIBUJAR
    // Activamos las texturas 2D.
    glEnable(GL_TEXTURE_2D);

    // Enlazamos la textura creada previamente.
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Actualizamos la textura con los colores del buffer de píxeles.
    glTexImage2D(GL_TEXTURE_2D,
            0,                // Nivel base de la textura.
            GL_RGBA,          // Formato interno (RGBA).
            WIDTH, HEIGHT,    // Dimensiones de la textura.
            0,                // Borde (debe ser 0).
            GL_RGBA,          // Formato de los datos.
            GL_UNSIGNED_BYTE, // Tipo de datos en el buffer.
            pixel_buffer);    // Datos del buffer.

    // Dibujamos un cuadrado cubriendo toda la ventana para mostrar la textura.
    glBegin(GL_QUADS);
    {
        // Definimos las coordenadas de textura y los vértices del cuadrado.
        glTexCoord2f(0, 1);
        glVertex3f(-1, -1, 0);

        glTexCoord2f(0, 0);
        glVertex3f(-1, 1, 0);

        glTexCoord2f(1, 0);
        glVertex3f(1, 1, 0);

        glTexCoord2f(1, 1);
        glVertex3f(1, -1, 0);
    }
    glEnd();  // Finalizamos el dibujo del cuadrado.

}

void loop() {
    // Establecemos el color de fondo de la ventana en negro.
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Iniciamos el bucle principal de la aplicación.
    while (!glfwWindowShouldClose(window)) {
        // Limpiamos el buffer de color y el buffer de profundidad.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Llamamos a la función `paint` para calcular y renderizar el conjunto de Mandelbrot.
        paint();

        // Intercambiamos los buffers de la ventana (doble buffering).
        glfwSwapBuffers(window);

        // Procesamos eventos pendientes de la ventana (como entradas del teclado o mouse).
        glfwPollEvents();
    }
}

void run() {
    // Inicializamos todos los recursos necesarios.
    init();

    // Ejecutamos el bucle principal.
    loop();

    // Finalizamos y liberamos recursos asociados con GLFW.
    glfwTerminate();
}

int main() {
    // Reservamos memoria para el buffer de píxeles.
    pixel_buffer = new GLuint[WIDTH * HEIGHT];

    // Imprimimos un mensaje de inicio utilizando la biblioteca fmt.
    fmt::print("Hola mundo 0.o fmt\n");

    // Llamamos a la función `run` para iniciar la aplicación.
    run();

    // Liberamos la memoria del buffer de píxeles.
    delete[] pixel_buffer;

    // Indicamos que el programa terminó correctamente.
    return 0;
}


