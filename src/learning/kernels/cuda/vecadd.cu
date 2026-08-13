#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// global marks this as a GPU kernel callable from the CPU
// a, b, c are pointers to GPU mem
__global__ void vectorAdd(float *a, float *b, float *c) {
    // each thread gets a unique id from this call
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int n = 8;
    // check how many bytes we need for 8 floats
    size_t bytes = n * sizeof(float);

    // allocate 8 floats worth of data, return the pointer
    // h signifies host
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // init the host arrays
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // allocate the device/gpu data
    // (void**)&d_a passes the address of
    // a pointer to cuda malloc
    // we don't really need (void**) in a 
    // .cu file
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes); 
    cudaMalloc((void**)&d_b, bytes); 
    cudaMalloc((void**)&d_c, bytes);  

    // copy the host data to the fevice 
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int success = 1;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            printf("Error at index %d: got %f, expected %f\n",
                    i, h_c[i], (h_a[i] + h_b[i]));
            success = 0;
            break;
        }
    }

    if (success) { printf("All elements are correct\n"); }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c); 

    return 0;
}

// general steps for a CUDA program
// 1. allocate mem on the CPU/host
// 2. init data on CPU
// 3. allocate mem on the GPU/device
// 4. Copy data from host to device
// 5. launch kernel to compute on device
// 6. copy results from device to host
// 7. verify results and free mem
