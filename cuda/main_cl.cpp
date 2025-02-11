/*//
// Created by fing.labcom on 22/01/2025.
//

#include <iostream>
#include <CL/cl.h>

std::string kernel_code =
    "__kernel void suma_opencl(__global float* a,__global float* b,__global float* c,const unsigned int size) {"
    "    int index = get_global_id(0);"
    "   if (index < size) {"
    "       c[index] = a[index] + b[index];"
    "    }"
    "}";

int main() {
    cl_platform_id* platform= nullptr;
    cl_uint num_plataforms=0;

    //-- obtener el numero de platadormas
    clGetPlatformIDs(0, nullptr, &num_plataforms);

    std::printf("Tot. plataformas:; %d\n", num_plataforms);

    auto platforms  = new cl_platform_id[num_plataforms];
    clGetPlatformIDs(num_plataforms, platforms, nullptr);

    for(int i=0;i<num_plataforms;i++) {
        char vendor[1024];
        char name[1024];
        char version[1024];


        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);

        std::printf("--------------------------------------------------\n");
        std::printf("Plataforma id=%d \n",i);
        std::printf("   Vendor: %s\n", vendor);
        std::printf("   Name: %s\n", name);
        std::printf("   Version: %s\n", version);

        std::printf("   ---devices\n");

        cl_uint num_devices=0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        //CL_DEVICE_TYPE_ALL
        //CL_DEVICE_TYPE_CPU
        //CL_DEVICE_TYPE_GPU

        std::printf("     num. devices: %d\n", num_devices);

        cl_device_id* devices= new cl_device_id[num_devices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);


        char device_name[1024];
        for(int di=0;di<num_devices;di++) {
            ;

            clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);

            std::printf("       Device id=%d  name: %s\n", di, device_name);
        }

    }

    //-------------------------------------------------------------------------
    std:printf("\n");
    std::printf("--------------------------------------------------------------------\n");

    cl_platform_id platform_id = platforms[4];
    cl_device_id device_id = nullptr; //device GPU
    cl_context context = nullptr; //necesitamos un contexto
    //se manda a una cola y se va ejecutando
    cl_command_queue commands_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_int error=0;


    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr); //sacamos solo 1, no necesitamos el ultimo argumento.

    char buffer[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    std::printf("Using device: %s\n", buffer);

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
    //queremos configurar algo? no
    //numero de dispositivos: 1
    // la referencia
    //

    commands_queue = clCreateCommandQueue(context, device_id, 0, &error);


    //--crear programa
    const char* src_program = kernel_code.c_str();
    program = clCreateProgramWithSource(context, 1, &src_program, nullptr, &error);
    if(program==nullptr) {
        //no pudo cargar el programa
        std::printf("Error al cargar el programa\n");
        exit(1);
    }

    error=clBuildProgram(program,1,&device_id, nullptr, nullptr, nullptr);
    if(error!=CL_SUCCESS) {
        size_t len;
        char buffer[1024];
        clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::printf("Build error: %s\n",buffer);
        exit(1);
    }

    std::string kernel_name = "suma_opencl";
    kernel  = clCreateKernel(program, "suma_opencl", &error);
    //recive el programa, el nombre del kernel, y el error

    if(kernel==nullptr) {
        std::printf("Can´t create kernel: '%s'\n",kernel_name.c_str());
        exit(1);
    }

    const size_t VECTOR_SIZE = 1024;

    float* h_A  = new float[VECTOR_SIZE];
    float* h_B  = new float[VECTOR_SIZE];
    float* h_C  = new float[VECTOR_SIZE];

    memset(h_C,0,VECTOR_SIZE*sizeof(float));
    for (int i=0 ; i<VECTOR_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    //--memeoria device
    size_t size_in_bytes=VECTOR_SIZE*sizeof(float);
    cl_mem d_A=clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);
    cl_mem d_B=clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);
    cl_mem d_C=clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);

    //--copiar: host-to-device
    clEnqueueWriteBuffer(commands_queue, d_A, CL_TRUE, 0, size_in_bytes, h_A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commands_queue, d_B, CL_TRUE, 0, size_in_bytes, h_B, 0, nullptr, nullptr);

    //--ejecutar el kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    //el primer algumento es el vector A, por eso arg_index=0
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(size_t), &VECTOR_SIZE);

    size_t global_work_size = VECTOR_SIZE;
    size_t local_work_size = 256;

    clEnqueueNDRangeKernel(
        commands_queue,
        kernel,
        1,
        0,
        &global_work_size,
        &local_work_size,
        0,
        nullptr,
        nullptr);

    //work_dim: tipo de grid, 2d, 3d o 1d.
    //...offset: indica que no me voy a dezplazat
    // global_word_size: el total de hilos que voy a utilizar
    // local_work_size: el numero de hilos que voy a utilizar
    // no eventos de espera,
    //no eventos de salida

    clFinish(commands_queue);

    //copiar: device-to-host
    clEnqueueReadBuffer(commands_queue, d_C, CL_TRUE, 0, size_in_bytes, h_C, 0, nullptr, nullptr);
    //leeemos desde dc, bloquemos hasta que termine, leemos en h_C, no hay offset, no hay eventos de espera, no hay eventos de salida

    for(int i=0;i<VECTOR_SIZE;i++) {
        std::printf("  %0.f",h_C[i]);
    }

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands_queue);

    return 0;
}*/


#include <iostream>
#include <CL/cl.h>

std::string kernel_code =
    "__kernel void suma_opencl(__global float *a, __global float *b, __global float *c, const unsigned int size) {"
    "    int index = get_global_id(0);"
    "    if (index < size) {"
    "        c[index] = a[index] + b[index];"
    "    }"
    "}";

int main() {
    // Listamos el número de dispositivos gráficos
    cl_platform_id *platforms = nullptr;
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::printf("No. plataformas: %d\n", numPlatforms);

    platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, nullptr);

    for (int i = 0; i < numPlatforms; i++) {
        char vendor[1024];
        char version[1024];
        char name[1024];

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);

        std::printf("Plataforma %d: %s\n\t %s - %s\n", i, vendor, version, name);
        std::printf("\t*** Dispositivos ***\n");

        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::printf("\tNo. : %d\n", numDevices);

        cl_device_id *devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, nullptr);

        for (int di = 0; di < numDevices; di++) {
            char deviceName[1024];
            clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            std::printf("\tId %d - Nombre: %s\n", di, deviceName);
        }
    }

    // Seleccionamos el dispositivo que vamos a usar (GPU)
    std::printf("\n---------------------------------------------\n");

    cl_platform_id platform_id = platforms[0]; // Es el índice de la plataforma nvidia
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue commands_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_int error = 0;

    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    char buffer[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    std::printf("Dispositivo seleccionado: %s\n", buffer);

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
    commands_queue = clCreateCommandQueue(context, device_id, 0, &error);

    // Creamos el programa
    const char *src_program = kernel_code.c_str();
    program = clCreateProgramWithSource(context, 1, &src_program, nullptr, &error);
    if (program == nullptr) {
        std::printf("Error al cargar el recurso\n");
        exit(1);
    }

    // Compilamos el programa
    error = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        size_t len = 0;
        char buffer[1024];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::printf("Error al compilar el programa: %s\n", buffer);
        exit(1);
    }

    std::string kernel_name = "suma_opencl";
    kernel = clCreateKernel(program, kernel_name.c_str(), &error);
    if (kernel == nullptr) {
        std::printf("Error al crear el kernel: %s\n", kernel_name.c_str());
        exit(1);
    }

    // Operaciones hostToDevice y viceversa
    // Allocate memory for the host vectors
    const size_t VECTOR_SIZE = 1024 * 1024 *100 ;
    float *h_A = new float[VECTOR_SIZE];
    float *h_B = new float[VECTOR_SIZE];
    float *h_C = new float[VECTOR_SIZE];
    memset(h_C, 0, VECTOR_SIZE * sizeof(float));

    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate memory for the device vectors
    size_t size_bytes = VECTOR_SIZE * sizeof(float);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);

    // Copy the host vectors to the device
    clEnqueueWriteBuffer(commands_queue, d_A, CL_TRUE, 0, size_bytes, h_A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commands_queue, d_B, CL_TRUE, 0, size_bytes, h_B, 0, nullptr, nullptr);

    // Invocamos el kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &VECTOR_SIZE);

    size_t global_work_size = VECTOR_SIZE;
    clEnqueueNDRangeKernel(commands_queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    clFinish(commands_queue);

    // Copiamos los resultados del dispositivo al host
    clEnqueueReadBuffer(commands_queue, d_C, CL_TRUE, 0, size_bytes, h_C, 0, nullptr, nullptr);

    for (int i = 0; i < 10; i++) {
        std::printf("%.0f ", h_C[i]);
    }

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands_queue);
    return 0;
}
