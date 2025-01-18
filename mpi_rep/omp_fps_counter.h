
#ifndef FPS_COUNTER_H
#define FPS_COUNTER_H

#include <chrono> // Incluye la biblioteca estándar para medir tiempo.

namespace ch = std::chrono; // Alias para simplificar el uso de std::chrono.

class omp_fps_counter {
private:
    int frames; // Contador de cuadros procesados desde el último cálculo de FPS.
    int fps;    // Almacena el número de cuadros por segundo calculado.
    ch::time_point<ch::high_resolution_clock> last_time; // Marca de tiempo del último cálculo de FPS.

public:
    omp_fps_counter(); // Constructor de la clase.

    void update(); // Método que actualiza el contador de cuadros y calcula los FPS si es necesario.

    int get_fps() const; // Devuelve el último valor calculado de FPS.
};

#endif // FPS_COUNTER_H
