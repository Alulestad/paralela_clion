//
// Created by Dami on 27/11/2024.
//

#ifndef FPS_COUNTER_H
#define FPS_COUNTER_H
//Son guardas de inclusión. Evitan que el archivo se incluya varias veces en el proyecto,
//lo que podría causar errores de compilación.
//Si el archivo ya fue incluido, el código entre estas directivas será ignorado


#include <chrono> // Incluye la biblioteca estándar para medir tiempo.

namespace ch = std::chrono; // Alias para simplificar el uso de std::chrono.

class fps_counter {
private:
    int frames; // Contador de cuadros procesados desde el último cálculo de FPS.
    int fps;    // Almacena el número de cuadros por segundo calculado.
    ch::time_point<ch::system_clock> last_time;
    // Marca de tiempo del último cálculo de FPS.

public:
    fps_counter(); // Constructor de la clase.

    void update(); // Método que actualiza el contador de cuadros y calcula los FPS si es necesario.

    int get_fps(); // Devuelve el último valor calculado de FPS.
};

#endif //FPS_COUNTER_H
