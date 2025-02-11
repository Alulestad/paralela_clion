
#ifndef FPS_COUNTER_H // Si FPS_COUNTER_H no está definido
#define FPS_COUNTER_H   // Entonces Lo definimos

#include <chrono> // Incluye la biblioteca estándar para medir tiempo.

namespace ch = std::chrono; // Alias para simplificar el uso de std::chrono.
// es un alias de espacio de nombres.
// :: resolución de ámbito es para acceder a elementos dentro de un espacio de nombres o clase
// este espacio hace referencia a variables, funciones, clases, etc.
// es util en caso de conflicto de nombres
// pueden haver namesspace anidados
class omp_fps_counter {
private: // Solo accesible dentro de la misma clase.
    // protected	Accesible dentro de la misma clase y en clases derivadas.
    // public	Accesible desde cualquier parte del código.
    int frames; // Contador de cuadros procesados desde el último cálculo de FPS.
    int fps;    // Almacena el número de cuadros por segundo calculado.
    ch::time_point<ch::high_resolution_clock> last_time; // Marca de tiempo del último cálculo de FPS.
    // time_point indica un punto de tiempo
    // high_resolution_clock es un reloj de alta resolución
    // <> permite definir el reloj que se requiere.
    // <> es un operador de plantilla que se utiliza para definir plantillas de clase o función.
    // es un tipo de dato vaiable en pocas palabras

public:
    omp_fps_counter(); // Constructor de la clase.

    void update(); // Metodo que actualiza el contador de cuadros y calcula los FPS si es necesario.

    int get_fps() const; // Devuelve el último valor calculado de FPS.
    //El const indica que este metodo no modifica ningún dato de la clase.
};

#endif // FPS_COUNTER_H
// // Fin de la protección contra inclusión múltiple
