// Created by Dami on 27/11/2024.

#include "omp_fps_counter.h" // Incluye el archivo de encabezado con la clase omp_fps_counter.
#include <fmt/core.h> // Incluye la biblioteca fmt para impresión formateada.

omp_fps_counter::omp_fps_counter() : frames(0), fps(0), last_time(ch::high_resolution_clock::now()) {
    // Inicialización de atributos mediante lista de inicialización.


    // Tambien se puede iniciar asi:
    // omp_fps_counter::omp_fps_counter() {
    //     frames = 0;  // Inicializa frames a 0.
    //     fps = 0;     // Inicializa fps a 0.
    //     last_time = ch::high_resolution_clock::now();  // Inicializa last_time con el tiempo actual.
    // }

    //con la diferencia que con lo puesto puedo hacer luego:
    //asignar un valor en el cuerpo del constructor
}
//estoy indicando que este constructor es parte de la clase omp_fps_counter.h
//omp_fps_counter() pertenece al ambito scope de la clase omp_fps_counter

int omp_fps_counter::get_fps() const {
    return fps; // Devuelve el último valor calculado de FPS.
}
//isntancio en el ambioto de la clase omp_fps_counter.h el metodo get_fps

void omp_fps_counter::update() {
    frames++; // Incrementa el número de cuadros procesados.

    auto current_time = ch::high_resolution_clock::now();
    // Captura el tiempo actual usando un relolj de alta resolución.

    ch::duration<double, std::milli> elapsed_time = current_time - last_time;
    // Calcula la duración (en milisegundos), especifico el tipo de dato, en millisegundos que requiero
    // y la resta de los tiempos actuales y el ultimo tiempo

    if (elapsed_time.count() > 1000.0) { // Verifica si ha pasado más de un segundo (1000 ms).
        fps = frames;               // Almacena el número de cuadros procesados como FPS.
        frames = 0;                 // Reinicia el contador de cuadros.
        last_time = current_time;   // Actualiza el tiempo de referencia.

        fmt::println("FPS: {}", fps); // Imprime el valor de FPS actual usando la biblioteca fmt.
    }
}
