#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <mpi.h>

#include "omp_fps_counter.h"

// Función para invertir los bytes de un entero de 32 bits
uint32_t _bswap32(uint32_t a) {
    return ((a & 0X000000FF) << 24) | ((a & 0X0000FF00) << 8) | ((a & 0x00FF0000) >> 8) | ((a & 0xFF000000) >> 24);
}

// Definición de constantes
#define WIDTH 1600
#define HEIGHT 900

const double x_max = 1;
const double x_min = -2;
const double y_min = -1;
const double y_max = 1;

const int max_iterations = 100;
const int PALETE_SIZE = 16;

const GLuint color_ramp[PALETE_SIZE] = {
    _bswap32(0xFFFF1010), _bswap32(0xFFF31017), _bswap32(0xFFE8101E), _bswap32(0xFFDC1126),
    _bswap32(0xFFD1112D), _bswap32(0xFFC51235), _bswap32(0xFFBA123C), _bswap32(0xFFAE1343),
    _bswap32(0xFFA3134B), _bswap32(0xFF971452), _bswap32(0xFF8C145A), _bswap32(0xFF801461),
    _bswap32(0xFF751568), _bswap32(0xFF691570), _bswap32(0xFF5E1677), _bswap32(0xFF54167D),
};

// Variables globales
static GLFWwindow* window = NULL;
GLuint textureID;
GLuint* pixel_buffer = nullptr;
omp_fps_counter fps;

// Función para inicializar las texturas de OpenGL
void initTextures() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Función para inicializar GLFW y configurar la ventana
void init() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        exit(-1);
    }

    window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGl C++", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window!" << std::endl;
        exit(-1);
    }

    glfwSetKeyCallback(window, [](GLFWwindow* window, auto key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    });

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    glfwMakeContextCurrent(window);
    std::string version = (char*)glGetString(GL_VERSION);
    std::string vendor = (char*)glGetString(GL_VENDOR);
    std::string render = (char*)glGetString(GL_RENDERER);

    fmt::print("OpenGL version supported {}\n", version);
    fmt::print("Vendor : {}\n", vendor);
    fmt::print("Renderer : {}\n", render);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glfwSwapInterval(1);
    initTextures();
}

// Función para calcular si un punto diverge en el conjunto de Mandelbrot
int divergente(double cx, double cy) {
    int iter = 0;
    double vx = cx;
    double vy = cy;

    while (iter < max_iterations && (vx * vx + vy * vy) <= 4.0) {
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
        return color_ramp[0];
    }

    return 0;
}

// Función para calcular el conjunto de Mandelbrot usando MPI
void mandelbrotMpi(int rank, int size, GLuint* local_pixel_buffer) {
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

    // Dividir el trabajo entre los procesos
    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_process;

    for (int i = 0; i < WIDTH; i++) {
        for (int j = start_row; j < end_row; j++) {
            double x = x_min + i * dx;
            double y = y_max - j * dy;
            int color = divergente(x, y);
            local_pixel_buffer[j * WIDTH + i] = color;
        }
    }
}

// Función para renderizar la imagen
void paint(GLuint* pixel_buffer) {
    fps.update();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0, 1); glVertex3f(-1, -1, 0);
        glTexCoord2f(0, 0); glVertex3f(-1, 1, 0);
        glTexCoord2f(1, 0); glVertex3f(1, 1, 0);
        glTexCoord2f(1, 1); glVertex3f(1, -1, 0);
    }
    glEnd();
}

// Función principal
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    GLuint* local_pixel_buffer = new GLuint[WIDTH * HEIGHT];

    if (rank == 0) {
        pixel_buffer = new GLuint[WIDTH * HEIGHT];
        init();
    }

    mandelbrotMpi(rank, size, local_pixel_buffer);

    // Recopilar los resultados en el proceso principal
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int start_row = i * (HEIGHT / size);
            int end_row = (i == size - 1) ? HEIGHT : start_row + (HEIGHT / size);
            MPI_Recv(&pixel_buffer[start_row * WIDTH], (end_row - start_row) * WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        memcpy(pixel_buffer, local_pixel_buffer, WIDTH * (HEIGHT / size) * sizeof(GLuint));
    } else {
        int start_row = rank * (HEIGHT / size);
        int end_row = (rank == size - 1) ? HEIGHT : start_row + (HEIGHT / size);
        MPI_Send(&local_pixel_buffer[start_row * WIDTH], (end_row - start_row) * WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            paint(pixel_buffer);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        delete[] pixel_buffer;
    }

    delete[] local_pixel_buffer;
    MPI_Finalize();

    return 0;
}