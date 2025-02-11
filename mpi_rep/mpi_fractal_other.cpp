#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <mpi.h>

// Mantener las funciones y constantes originales
uint32_t _bswap32(uint32_t a) {
    return ((a & 0X000000FF) << 24) |
           ((a & 0X0000FF00) << 8) |
           ((a & 0x00FF0000) >> 8) |
           ((a & 0xFF000000) >> 24);
}

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

static GLFWwindow* window = NULL;
GLuint textureID;
GLuint* pixel_buffer = nullptr;
GLuint* local_buffer = nullptr;  // Buffer local para cada proceso

// Función Mandelbrot sin cambios
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

// Nueva implementación con MPI
void mandelbrotMPI(int rank, int size) {
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

    // Calcular cuántas filas procesará cada proceso
    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_process;

    // Calcular tamaño del buffer local
    int local_size = (end_row - start_row) * WIDTH;
    local_buffer = new GLuint[local_size];

    // Computar la porción local del fractal
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = x_min + i * dx;
            double y = y_max - j * dy;
            int color = divergente(x, y);
            local_buffer[(j - start_row) * WIDTH + i] = color;
        }
    }

    // Reunir resultados en el proceso 0
    if (rank == 0) {
        // Copiar datos locales al buffer final
        memcpy(pixel_buffer + start_row * WIDTH, local_buffer, local_size * sizeof(GLuint));

        // Recibir datos de otros procesos
        for (int i = 1; i < size; i++) {
            int recv_start = i * rows_per_process * WIDTH;
            int recv_size = (i == size - 1) ?
                           (HEIGHT - i * rows_per_process) * WIDTH :
                           rows_per_process * WIDTH;

            MPI_Recv(pixel_buffer + recv_start,
                    recv_size,
                    MPI_UNSIGNED,
                    i,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
        }
    } else {
        // Enviar datos locales al proceso 0
        MPI_Send(local_buffer,
                local_size,
                MPI_UNSIGNED,
                0,
                0,
                MPI_COMM_WORLD);
    }

    delete[] local_buffer;
}

void initTextures() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void init(int rank) {
    if (rank == 0) {  // Solo el proceso 0 inicializa GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "MPI Mandelbrot", NULL, NULL);
        if (!window) {
            glfwTerminate();
            std::cerr << "Failed to create GLFW window!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        glfwSetKeyCallback(window, [](GLFWwindow* window, auto key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        });

        glfwMakeContextCurrent(window);
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

void paint(int rank) {
    if (rank == 0) {  // Solo el proceso 0 realiza el renderizado
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
}

void loop(int rank, int size) {
    if (rank == 0) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    }

    while (rank == 0 ? !glfwWindowShouldClose(window) : true) {
        // Sincronizar todos los procesos
        MPI_Barrier(MPI_COMM_WORLD);

        mandelbrotMPI(rank, size);

        if (rank == 0) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            paint(rank);
            glfwSwapBuffers(window);
            glfwPollEvents();

            if (glfwWindowShouldClose(window)) {
                // Notificar a otros procesos que deben terminar
                int terminate = 1;
                for (int i = 1; i < size; i++) {
                    MPI_Send(&terminate, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                }
                break;
            } else {
                // Notificar a otros procesos que continúen
                int continue_flag = 0;
                for (int i = 1; i < size; i++) {
                    MPI_Send(&continue_flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            // Procesos esclavos esperan señal de terminación
            int terminate_flag;
            MPI_Recv(&terminate_flag, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (terminate_flag) break;
        }
    }
}

void run(int rank, int size) {
    init(rank);
    loop(rank, size);
    if (rank == 0) {
        glfwTerminate();
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        pixel_buffer = new GLuint[WIDTH * HEIGHT];
        fmt::print("Iniciando MPI Mandelbrot con {} procesos\n", size);
    }

    run(rank, size);

    if (rank == 0) {
        delete[] pixel_buffer;
    }

    MPI_Finalize();
    return 0;
}