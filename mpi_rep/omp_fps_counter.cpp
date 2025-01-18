// Created by Dami on 27/11/2024.

#include "omp_fps_counter.h" // Incluye el archivo de encabezado con la clase omp_fps_counter.
#include <fmt/core.h> // Incluye la biblioteca fmt para impresión formateada.

omp_fps_counter::omp_fps_counter() : frames(0), fps(0), last_time(ch::high_resolution_clock::now()) {
    // Inicialización de atributos mediante lista de inicialización.
}

int omp_fps_counter::get_fps() const {
    return fps; // Devuelve el último valor calculado de FPS.
}

void omp_fps_counter::update() {
    frames++; // Incrementa el número de cuadros procesados.

    auto current_time = ch::high_resolution_clock::now();
    // Captura el tiempo actual.

    ch::duration<double, std::milli> elapsed_time = current_time - last_time;
    // Calcula la duración (en milisegundos) desde el último cálculo de FPS.

    if (elapsed_time.count() > 1000.0) { // Verifica si ha pasado más de un segundo (1000 ms).
        fps = frames;               // Almacena el número de cuadros procesados como FPS.
        frames = 0;                 // Reinicia el contador de cuadros.
        last_time = current_time;   // Actualiza el tiempo de referencia.

        fmt::println("FPS: {}", fps); // Imprime el valor de FPS actual usando la biblioteca fmt.
    }
}
