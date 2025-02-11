#ifndef FPS_COUNTER_H
#define FPS_COUNTER_H

#include <chrono> // Biblioteca para medir tiempo

namespace ch = std::chrono; // Alias para simplificar el uso de std::chrono

class fps_counter {
private:
    int frames; // Contador de cuadros procesados
    int fps;    // FPS calculado
    ch::high_resolution_clock::time_point last_time; // Marca de tiempo del último cálculo

public:
    fps_counter(); // Constructor
    void update(); // Actualiza el contador de cuadros y calcula FPS
    int get_fps(); // Devuelve el FPS calculado
};

#endif // FPS_COUNTER_H
