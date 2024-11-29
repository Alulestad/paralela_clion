//
// Created by Dami on 27/11/2024.
//

#include "fps_counter.h" // Incluye la declaración de la clase.
#include <fmt/core.h>    // Incluye la biblioteca fmt para impresión formateada.

fps_counter::fps_counter() {
    // Inicializa los atributos de la clase.
    frames = 0; // No se han procesado cuadros inicialmente.
    fps = 0;    // El FPS inicial es 0.
    last_time = ch::high_resolution_clock::now();
    // Guarda el tiempo actual como referencia para cálculos futuros.
}

int fps_counter::get_fps() {
    // Devuelve el último valor de FPS calculado.
    return this->fps;
}

void fps_counter::update() {
    frames++; // Incrementa el número de cuadros procesados.

    auto current_time = ch::high_resolution_clock::now();
    // Captura el tiempo actual.

    ch::duration<double, std::milli> tiempo = current_time - last_time;
    // Calcula la duración (en milisegundos) desde el último cálculo de FPS.

    if (tiempo.count() > 1000) {
        // Si ha pasado más de un segundo (1000 ms):
        fps = frames;               // Almacena el número de cuadros procesados como FPS.
        frames = 0;                 // Reinicia el contador de cuadros.
        last_time = current_time;   // Actualiza el tiempo de referencia.

        fmt::println("FPS: {}", fps);
        // Imprime el valor de FPS actual usando la biblioteca fmt.
    }
}
