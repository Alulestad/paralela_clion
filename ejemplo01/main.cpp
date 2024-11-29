#include <iostream>
#include <fmt/core.h>
#include <GLFW/glfw3.h>

static GLFWwindow* window=nullptr;

void init() {
    /* Initialize the library */
    if (!glfwInit())
        exit(-1);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "OpenGL C++", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    });

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /*
    std::string version = (char *)glGetString(GL_VERSION);
    std::string vendor = (char *)glGetString(GL_VENDOR);
    std::string render = (char *)glGetString(GL_RENDER);
    */

    fmt::print("##################################################################\n");
    fmt::print("#########> PROGRAMACION PARALELA - ING. JAIME SALVADOR <##########\n");
    fmt::print("######################> DANIEL LLUMIQUINGA <######################\n");
    fmt::print("##################################################################\n");

    const char* versionC = (const char*)glGetString(GL_VERSION);
    const char* vendorC = (const char*)glGetString(GL_VENDOR);
    const char* renderC = (const char*)glGetString(GL_RENDER);

    if (versionC) {
        fmt::print("OpenGL version: {}\n", versionC);
    } else {
        fmt::print("Failed to retrieve OpenGL version.\n");
    }

    if (vendorC) {
        fmt::print("OpenGL vendor: {}\n", vendorC);
    } else {
        fmt::print("Failed to retrieve OpenGL vendor.\n");
    }

    if (renderC) {
        fmt::print("OpenGL render: {}\n", renderC);
    } else {
        fmt::print("Failed to retrieve OpenGL render.\n");
    }


    glMatrixMode(GL_PROJECTION);//Configura una proyección ortogonal
    // esta matriz transforma el espacio del mundo a un espacio que se puede proyectar en la pantalla.
    // Define cómo se transforman las coordenadas 3D de los objetos al espacio 2D de la pantalla
    glLoadIdentity();//matriz identidad (es decir, una matriz sin ninguna transformación aplicada)
    // Representa una transformación "en blanco", sin ningún escalado, rotación o traslación
    glOrtho(-1,1,-1,1,-1,1); //Configura una proyección ortográfica
    //significa que los objetos se proyectarán sin perspectiva,
    // es decir, los objetos mantendrán su tamaño independientemente de su distancia de la cámara.
    //define los rangos de eje x (izquierda derecha), y(arriba abajo) y z (profundidad)
    glMatrixMode(GL_MODELVIEW);//Cambia el modo de matriz a matriz de vista del modelo.
    //las operaciones siguientes (como glLoadIdentity) deben aplicarse a la matriz de vista del modelo.
    //Esta matriz se utiliza para transformar los objetos del espacio local (sus propias coordenadas)
    // al espacio global (coordenadas del mundo) y para posicionar la "cámara" o punto de vista en la escena
    glLoadIdentity();
    //Restaura la matriz de vista del modelo a la identidad,
    // eliminando cualquier transformación acumulada en la matriz de vista del modelo.
    //sin el glMatrixMode(GL_MODELVIEW);, glLoadIdentity(); afectaría la matriz de proyección (o cualquier otra que estuviera activa).


    // Enable v-sync es el número de veces por segundo que la pantalla se actualiza de arriba hacia abajo
    glfwSwapInterval(1);
    // ayuda a evitar problemas visuales como el "screen tearing"



}

void paint() {
    glBegin(GL_TRIANGLES);
        glVertex2d(-1.0f, -1.0f); // Vértice 1
        glVertex2d(0.0f, 0.0f);  // Vértice 2
        glVertex2d(0.0f, -1.0f);   // Vértice 3
    glEnd();
}

void loop() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); //color de fondo negro

    while ( !glfwWindowShouldClose(window) ) { //Mientras la ventana esté abierta
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Limpia el framebuffer usando el color de fondo y el buffer de profundidad.

        paint();

        glfwSwapBuffers(window); //Intercambia los buffers para mostrar la imagen.

        glfwPollEvents();//Procesa eventos de la ventana, como teclas y redimensionamiento.
    }

}

void run() {
    init();
    loop();

    glfwTerminate();
}

int main() {
    //std::cout << "Hello, World!" << std::endl;

    fmt::println("Hola mundo 0.o fmt");

    run();

    return 0;
}
