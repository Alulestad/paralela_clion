#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <mpi.h>
#include <vector>
#include "fps_counter.h"


// GRUPO 1
//INTEGRANTES: COLOMA DILLAN
//             COYAGO HENRY
//             ESTRELLA JUAN

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
// Definición de constantes
#define WIDTH 1280
#define HEIGHT 720

const double x_max = 1;
const double x_min = -2;
const double y_min = -1;
const double y_max = 1;

const int max_iterations = 100;
const int PALETE_SIZE = 16;

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
// Variables globales
static GLFWwindow* window = NULL;
GLuint textureID;
GLuint* pixel_buffer = nullptr;
fps_counter fps;
int rank, size;  // Variables MPI para rango y número de procesos

//--------------------------------------------------------
void initTextures() {
    if (rank == 0) {  // Solo el proceso principal inicializa OpenGL
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                WIDTH, HEIGHT, 0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                NULL
        );

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

//--------------------------------------------------------
void init() {
    if (rank == 0) {  // Solo el proceso principal inicializa la ventana
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL C++ MPI Grupal", NULL, NULL);
        if (!window) {
            glfwTerminate();
            std::cerr << "Failed to create GLFW window!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        glfwSetKeyCallback(window, [](GLFWwindow* window, auto key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        });

        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
            glViewport(0, 0, width, height);
        });

        glfwMakeContextCurrent(window);

        std::string version = (char *)glGetString(GL_VERSION);
        std::string vendor = (char *)glGetString(GL_VENDOR);
        std::string render = (char *)glGetString(GL_RENDERER);

        fmt::print("OpenGL version supported {}\n", version);
        fmt::print("Vendor : {}\n", vendor);
        fmt::print("Renderer : {}\n", render);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1,1,-1,1,-1,1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glfwSwapInterval(1);

        initTextures();
    }
}

//--------------------------------------------------------
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

void mandelbrotMPI() {
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

    // Calcular cuántas filas procesará cada proceso
    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_process;

    // Buffer local para cada proceso
    std::vector<GLuint> local_buffer((end_row - start_row) * WIDTH);

    // Calcular porción local del conjunto de Mandelbrot
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = x_min + i * dx;
            double y = y_max - j * dy;
            int color = divergente(x, y);
            local_buffer[(j - start_row) * WIDTH + i] = color;
        }
    }

    // Recolectar resultados en el proceso principal
    if (rank == 0) {
        // Copiar datos locales al buffer principal
        std::copy(local_buffer.begin(), local_buffer.end(),
                 pixel_buffer + start_row * WIDTH);

        // Recibir datos de otros procesos
        for (int r = 1; r < size; r++) {
            int r_start = r * rows_per_process;
            int r_end = (r == size - 1) ? HEIGHT : r_start + rows_per_process;
            int r_size = (r_end - r_start) * WIDTH;

            MPI_Recv(pixel_buffer + r_start * WIDTH,
                    r_size,
                    MPI_UNSIGNED,
                    r,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
        }
    } else {
        // Enviar datos locales al proceso principal
        MPI_Send(local_buffer.data(),
                local_buffer.size(),
                MPI_UNSIGNED,
                0,
                0,
                MPI_COMM_WORLD);
    }
}

//--------------------------------------------------------
void paint() {
    if (rank == 0) {
        fps.update();
    }

    mandelbrotMPI();

    if (rank == 0) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(GL_TEXTURE_2D,
                0,
                GL_RGBA,
                WIDTH, HEIGHT,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                pixel_buffer);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0, 1);
            glVertex3f(-1, -1, 0);

            glTexCoord2f(0, 0);
            glVertex3f(-1, 1, 0);

            glTexCoord2f(1, 0);
            glVertex3f(1, 1, 0);

            glTexCoord2f(1, 1);
            glVertex3f(1, -1, 0);
        }
        glEnd();
    }
}

void loop() {
    if (rank == 0) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Enviar señal a todos los procesos para calcular
            int continue_flag = 1;
            for (int r = 1; r < size; r++) {
                MPI_Send(&continue_flag, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
            }

            paint();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        // Enviar señal de terminación a todos los procesos
        int stop_flag = 0;
        for (int r = 1; r < size; r++) {
            MPI_Send(&stop_flag, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
        }
    } else {
        // Procesos trabajadores esperan señales del proceso principal
        while (true) {
            int flag;
            MPI_Recv(&flag, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (!flag) break;  // Terminar si se recibe señal de parada

            paint();  // Calcular su parte del conjunto
        }
    }
}

void run() {
    init();
    loop();

    if (rank == 0) {
        glfwTerminate();
    }
}

int main(int argc, char** argv) {
    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        pixel_buffer = new GLuint[WIDTH * HEIGHT];
        fmt::print("Iniciando Mandelbrot con MPI (Procesos: {})\n", size);
    }

    run();

    if (rank == 0) {
        delete[] pixel_buffer;
    }

    MPI_Finalize();
    return 0;
}