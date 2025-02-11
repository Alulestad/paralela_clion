#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <mpi.h>
#include <GL/freeglut.h>
#include "omp_fps_counter.h"

//--------------------------------------------------------
// Función para invertir los bytes de un entero de 32 bits
uint32_t _bswap32(uint32_t a) {
    return ((a & 0X000000FF) << 24) |
           ((a & 0X0000FF00) << 8) |
           ((a & 0x00FF0000) >> 8) |
           ((a & 0xFF000000) >> 24);
}

//--------------------------------------------------------
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
    _bswap32(0xFFFF1010), _bswap32(0xFFF31017),
    _bswap32(0xFFE8101E), _bswap32(0xFFDC1126),
    _bswap32(0xFFD1112D), _bswap32(0xFFC51235),
    _bswap32(0xFFBA123C), _bswap32(0xFFAE1343),
    _bswap32(0xFFA3134B), _bswap32(0xFF971452),
    _bswap32(0xFF8C145A), _bswap32(0xFF801461),
    _bswap32(0xFF751568), _bswap32(0xFF691570),
    _bswap32(0xFF5E1677), _bswap32(0xFF54167D),
};

//--------------------------------------------------------
// Variables globales
static GLFWwindow* window = NULL;
GLuint textureID;
GLuint* pixel_buffer = nullptr;
GLuint* local_buffer = nullptr;  // Buffer local para cada proceso MPI
omp_fps_counter fps;

// Variables MPI
int rank, size;
int rows_per_process;
int extra_rows;

//--------------------------------------------------------
void initTextures() {
    if (rank == 0) {  // Solo el proceso principal inicializa OpenGL
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

//--------------------------------------------------------
void init() {
    if (rank == 0) {  // Solo el proceso principal inicializa GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL MPI", NULL, NULL);
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

    // Calcular las filas que procesará cada proceso
    int start_row = rank * rows_per_process + std::min(rank, extra_rows);
    int num_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    // Procesar las filas asignadas
    for (int j = 0; j < num_rows; j++) {
        int global_j = start_row + j;
        for (int i = 0; i < WIDTH; i++) {
            double x = x_min + i * dx;
            double y = y_max - global_j * dy;
            local_buffer[j * WIDTH + i] = divergente(x, y);
        }
    }

    // Recopilar resultados usando MPI_Gather
    if (rank == 0) {
        MPI_Gather(local_buffer, num_rows * WIDTH, MPI_UNSIGNED,
                  pixel_buffer, rows_per_process * WIDTH, MPI_UNSIGNED,
                  0, MPI_COMM_WORLD);

        // Recoger las filas extra si existen
        if (extra_rows > 0) {
            int offset = size * rows_per_process * WIDTH;
            for (int i = 0; i < extra_rows; i++) {
                MPI_Recv(&pixel_buffer[offset + i * WIDTH],
                        WIDTH, MPI_UNSIGNED, i,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Gather(local_buffer, num_rows * WIDTH, MPI_UNSIGNED,
                  NULL, 0, MPI_UNSIGNED,
                  0, MPI_COMM_WORLD);

        // Enviar filas extra si corresponde
        if (rank < extra_rows) {
            MPI_Send(&local_buffer[rows_per_process * WIDTH],
                    WIDTH, MPI_UNSIGNED,
                    0, 0, MPI_COMM_WORLD);
        }
    }
}

//--------------------------------------------------------
void renderText(const char* text, float x, float y) {
    glDisable(GL_TEXTURE_2D);
    glRasterPos2f(x, y);

    while (*text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *text);
        text++;
    }
    glEnable(GL_TEXTURE_2D);
}

void paint() {
    if (rank == 0) {
        fps.update();
    }

    mandelbrotMPI();

    if (rank == 0) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0, 1); glVertex3f(-1, -1, 0);
            glTexCoord2f(0, 0); glVertex3f(-1, 1, 0);
            glTexCoord2f(1, 0); glVertex3f(1, 1, 0);
            glTexCoord2f(1, 1); glVertex3f(1, -1, 0);
        }
        glEnd();

        // Mostrar FPS en pantalla
        char fps_text[32];
        snprintf(fps_text, sizeof(fps_text), "FPS: %d", fps.get_fps());
        renderText(fps_text, -0.95f, 0.9f);  // Posición en la esquina superior izquierda
    }
}


void loop() {
    if (rank == 0) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            paint();

            glfwSwapBuffers(window);
            glfwPollEvents();

            // Notificar a otros procesos si la ventana se cierra
            int should_continue = !glfwWindowShouldClose(window);
            MPI_Bcast(&should_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } else {
        int should_continue = 1;
        while (should_continue) {
            paint();
            MPI_Bcast(&should_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
    glutInit(&argc, argv);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calcular distribución de trabajo
    rows_per_process = HEIGHT / size;
    extra_rows = HEIGHT % size;

    // Calcular tamaño del buffer local
    int local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);
    local_buffer = new GLuint[local_rows * WIDTH];

    if (rank == 0) {
        pixel_buffer = new GLuint[WIDTH * HEIGHT];
        fmt::print("Iniciando Mandelbrot MPI con {} procesos\n", size);
    }

    run();

    delete[] local_buffer;
    if (rank == 0) {
        delete[] pixel_buffer;
    }

    MPI_Finalize();
    return 0;
}