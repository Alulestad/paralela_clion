#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <omp.h>

//--------------------------------------------------------
// Función para invertir los bytes de un entero de 32 bits
uint32_t _bswap32(uint32_t a) {
    return
        ((a & 0X000000FF) << 24) |
        ((a & 0X0000FF00) <<  8) |
        ((a & 0x00FF0000) >>  8) |
        ((a & 0xFF000000) >> 24);
}

//--------------------------------------------------------
// Definición de macros y constantes
#define WIDTH 1600      // Ancho de la ventana
#define HEIGHT 900      // Altura de la ventana
#define PALETTE_SIZE 16 // Tamaño de la paleta de colores

// Paleta de colores en formato RGBA
const GLuint color_ramp[PALETTE_SIZE] = {
    _bswap32(0xFFFF1010), _bswap32(0xFFF31017), _bswap32(0xFFE8101E), _bswap32(0xFFDC1126),
    _bswap32(0xFFD1112D), _bswap32(0xFFC51235), _bswap32(0xFFBA123C), _bswap32(0xFFAE1343),
    _bswap32(0xFFA3134B), _bswap32(0xFF971452), _bswap32(0xFF8C145A), _bswap32(0xFF801461),
    _bswap32(0xFF751568), _bswap32(0xFF691570), _bswap32(0xFF5E1677), _bswap32(0xFF54167D),
};

//--------------------------------------------------------
// Variables globales
static GLFWwindow* window = NULL;
GLuint textureID;
GLuint* pixel_buffer = nullptr;

//--------------------------------------------------------
// Función para inicializar texturas
void initTextures() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

//--------------------------------------------------------
// Inicialización de GLFW
void initGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        exit(-1);
    }

    window = glfwCreateWindow(WIDTH, HEIGHT, "Gradiente OpenGL", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window!" << std::endl;
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
        glViewport(0, 0, width, height);
    });

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glfwSwapInterval(1);
    initTextures();
}

//--------------------------------------------------------
// Función para generar el gradiente horizontal
void generateGradient() {
    #pragma omp parallel for
    for (int idx = 0; idx < WIDTH * HEIGHT; idx++) {
        int i = idx % WIDTH;                      // Coordenada X
        int j = idx / WIDTH;                      // Coordenada Y
        int index = i * PALETTE_SIZE / WIDTH;     // Cálculo del índice de color
        pixel_buffer[j * WIDTH + i] = color_ramp[PALETTE_SIZE - index - 1];
    }
}

//--------------------------------------------------------
// Renderizado
void renderFrame() {
    generateGradient();  // Generar gradiente

    // Actualizar textura con los datos del buffer de píxeles
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer);

    // Dibujar un cuadrado con la textura
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex3f(-1, -1, 0);
    glTexCoord2f(0, 0); glVertex3f(-1, 1, 0);
    glTexCoord2f(1, 0); glVertex3f(1, 1, 0);
    glTexCoord2f(1, 1); glVertex3f(1, -1, 0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}

//--------------------------------------------------------
// Bucle principal
void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        renderFrame();  // Renderizar el gradiente
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

//--------------------------------------------------------
// Función principal
int main() {
    // Inicializar buffer de píxeles
    pixel_buffer = new GLuint[WIDTH * HEIGHT];

    fmt::print("Inicializando Gradiente con OpenGL y OpenMP...\n");
    initGLFW();
    mainLoop();

    // Liberar memoria
    delete[] pixel_buffer;
    glfwTerminate();
    return 0;
}
