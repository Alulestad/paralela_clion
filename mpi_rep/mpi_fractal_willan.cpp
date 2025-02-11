#include <iostream>
#include <mpi.h>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>  // Necesario para las funciones de renderizado de texto

#define WIDTH 1600
#define HEIGHT 900

// Parámetros que se compartirán via Bcast
struct MandelbrotParams {
    double x_min, x_max, y_min, y_max;
    int maxIteration;
}; //en C++ esto es una estructura que permite almacenar variables para un manejo mas limpio y ordenado.

MandelbrotParams params; // Estructura para parámetros globales
const int PALETTE_SIZE = 16;

uint32_t _bswap32(uint32_t a) {
    return ((a & 0X000000FF) << 24) | ((a & 0X0000FF00) << 8) | ((a & 0x00FF0000) >> 8) | ((a & 0xFF000000) >> 24);
}

const GLuint color_ramp[] = {
    0xFFFF1010, 0xFFEE0E1B, 0xFFDD0D27, 0xFFCC0C32, 0xFFBB0B3E, 0xFFAA0A4A,
    0xFF990955, 0xFF880861, 0xFF77076C, 0xFF660678, 0xFF550584, 0xFF44048F,
    0xFF33039B, 0xFF2202A6, 0xFF1101B2, 0xFF0000BE
};

static GLFWwindow* window = nullptr;
GLuint textureID;
GLuint* pixel_buffer = nullptr;

// Variables para contar los FPS
double lastTime = 0.0;
int frameCount = 0;

// Función para calcular el Mandelbrot (usando parámetros broadcasted)
static void mandelbrotCpu(int rank, int rows_per_process, GLuint* local_pixels) {
    //rank: hace referencia al proceso el cual se va a desarrollar.
    //rows_per_process: son las filas para cada proceso.
    //local_pixels: es el buffer local para cada proceso.

    int black = 0x000000;

    for (int px = 0; px < WIDTH; px++) {
        for (int py = 0; py < rows_per_process; py++) {
            int global_py = rank * rows_per_process + py;
            double x0 = params.x_min + (params.x_max - params.x_min) * px / WIDTH;
            double y0 = params.y_min + (params.y_max - params.y_min) * global_py / HEIGHT;
            double x = 0.0;
            double y = 0.0;
            int iteration = 0;
            double R = 2.0;
            while (x * x + y * y <= R * R && iteration < params.maxIteration) {
                double tempX = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = tempX;
                iteration++;
            }
            local_pixels[py * WIDTH + px] = (iteration == params.maxIteration) ? black : color_ramp[iteration % PALETTE_SIZE];
        }
    }
}

static void paint(int rank, int rows_per_process, GLuint* local_pixels) {
    mandelbrotCpu(rank, rows_per_process, local_pixels);
}

static void initTextures() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void renderText(const std::string& text, float x, float y) {
    glRasterPos2f(x, y);
    for (char c : text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }
}

static void loop(int rank, int size, GLuint* local_pixels, int rows_per_process) {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Color de fondo negro
    frameCount = 0; // Inicializar contador de frames
    lastTime = glfwGetTime(); // Guardar el tiempo inicial
    double fps = 0.0; // Variable para almacenar el FPS más reciente

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        frameCount++;

        // Limpiar la pantalla en cada iteración del bucle
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calcular la parte del Mandelbrot
        paint(rank, rows_per_process, local_pixels);

        // Recolectar resultados de cada rank en el proceso 0
        MPI_Gather(local_pixels, rows_per_process * WIDTH, MPI_UNSIGNED,
                   pixel_buffer, rows_per_process * WIDTH, MPI_UNSIGNED,
                   0, MPI_COMM_WORLD);
        // local_pixels: Puntero al buffer de envío (contiene los datos locales de cada proceso).
        // pixel_buffer: Puntero al buffer de recepción en el proceso raíz (rank 0).
        // 0: El proceso raíz que va a recibir los datos (rank 0).

        // Solo dibujar el fractal si es el proceso 0
        if (rank == 0) {
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

            // Actualizar FPS cada segundo
            if (currentTime - lastTime >= 1.0) {
                fps = frameCount / (currentTime - lastTime); // Calcular FPS
                frameCount = 0;
                lastTime = currentTime;
            }

            // Renderizar FPS y maxIteration en la parte superior izquierda en cada frame
            std::string infoText = fmt::format("FPS: {:.2f}      Max Iterations: {}", fps, params.maxIteration);
            glDisable(GL_TEXTURE_2D);  // Deshabilitar texturas para renderizar solo el texto
            renderText(infoText, -0.9f, 0.9f); // Posicionamos el texto en la parte superior izquierda
        }

        glfwSwapBuffers(window); // Intercambiar buffers
        glfwPollEvents(); // Procesar eventos de la ventana
    }
}

static void init(int rank) {
    // Inicializar GLUT
    int argc = 1;
    char* argv[] = {(char*)"program"};
    glutInit(&argc, argv);  // Debes inicializar GLUT antes de usar cualquier función de GLUT

    if (!glfwInit()) exit(-1);

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot MPI", NULL, NULL);
    if (!window) throw std::runtime_error("Failed to create the GLFW window");

    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
            glfwSetWindowShouldClose(window, true);
    });

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidmode = glfwGetVideoMode(primaryMonitor);
    glfwSetWindowPos(window, (vidmode->width - WIDTH) / 2, (vidmode->height - HEIGHT) / 2);

    glfwMakeContextCurrent(window);
    fmt::println("OpenGL Version: {}", (char*)glGetString(GL_VERSION));
    fmt::println("OpenGL Vendor: {}", (char*)glGetString(GL_VENDOR));
    fmt::println("OpenGL Renderer: {}", (char*)glGetString(GL_RENDERER));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glfwSwapInterval(1);
    glfwShowWindow(window);

    initTextures();
}

static void run(int rank, int size) {
    int rows_per_process = HEIGHT / size;
    GLuint* local_pixels = new GLuint[rows_per_process * WIDTH];

    // Inicializar parámetros en el proceso 0 y broadcastearlos
    if (rank == 0) {
        params = { -2.0, 1.0, -1.0, 1.0, 100 }; // x_min, x_max, y_min, y_max, maxIteration
        pixel_buffer = new GLuint[WIDTH * HEIGHT];
    }

    // Broadcast de los parámetros a todos los procesos
    MPI_Bcast(&params, sizeof(params), MPI_BYTE, 0, MPI_COMM_WORLD);
    //difundir (broadcast) datos desde un proceso raíz a todos los demás procesos
    //&params → Dirección del dato a enviar
    //sizeof(params) → Tamaño del dato a enviar
    //MPI_BYTE → Tipo de dato a enviar
    //0 → Proceso raíz que envía los datos
    //MPI_COMM_WORLD → Comunicador
    //En pocas palabras: ahora todos inicializarion la estructura params.


    if (rank == 0) {
        init(rank);
        loop(rank, size, local_pixels, rows_per_process);
    } else {
        while (true) {
            mandelbrotCpu(rank, rows_per_process, local_pixels);
            MPI_Gather(local_pixels, rows_per_process * WIDTH, MPI_UNSIGNED,
                       nullptr, 0, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

            // recolecta datos desde múltiples procesos y los envía a un proceso raíz

        }
    }

    delete[] local_pixels;
    if (rank == 0) delete[] pixel_buffer;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        run(rank, size);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}