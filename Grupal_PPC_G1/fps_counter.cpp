#include "fps_counter.h"
#include <fmt/core.h> // Biblioteca fmt para impresión formateada

// Constructor: inicializa variables y registra el tiempo actual
fps_counter::fps_counter() : frames(0), fps(0), last_time(ch::high_resolution_clock::now()) {}

int fps_counter::get_fps() {
    return fps; // Devuelve el FPS calculado
}

void fps_counter::update() {
    frames++; // Incrementa el número de cuadros procesados

    auto current_time = ch::high_resolution_clock::now(); // Captura el tiempo actual
    ch::duration<double, std::milli> tiempo = current_time - last_time; // Calcula el tiempo transcurrido

    if (tiempo.count() > 1000) { // Si ha pasado más de un segundo (1000 ms)
        fps = frames;            // Guarda el número de cuadros procesados como FPS
        frames = 0;              // Reinicia el contador de cuadros
        last_time = current_time; // Actualiza el tiempo de referencia

        fmt::println("FPS: {}", fps); // Imprime el FPS actual
    }
}
