#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda_runtime.h>

/*
SETUP: 2D Lid-Driven Cavity (GPU)
*/

#define IDX(i, j) ((i) * N_ext + (j))

// Error check macro - wrap every CUDA call with this
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// GHOST NODE KERNELS
__global__ void updateVelocityGhostsKernel(double* u, double* v, int N, int N_ext) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (k <= N) {
        // TOP WALL (Row N+1: Moving Lid)
        u[IDX(N + 1, k)] = 2.0 * 1.0 - u[IDX(N, k)];
        v[IDX(N + 1, k)] = 2.0 * 0.0 - v[IDX(N, k)];

        // BOTTOM WALL (Row 0)
        u[IDX(0, k)] = -u[IDX(1, k)];
        v[IDX(0, k)] = -v[IDX(1, k)];

        // LEFT WALL (Col 0)
        u[IDX(k, 0)] = -u[IDX(k, 1)];
        v[IDX(k, 0)] = -v[IDX(k, 1)];

        // RIGHT WALL (Col N+1)
        u[IDX(k, N + 1)] = -u[IDX(k, N)];
        v[IDX(k, N + 1)] = -v[IDX(k, N)];
    }
}

__global__ void pinPressureKernel(double* p, int N_ext) {
    p[IDX(1, 1)] = 0.0;
}

// PREDICTOR KERNEL
__global__ void predictorKernel(double* u, double* v, double* u_star, double* v_star, int N, int N_ext, double h, double dt, double nu) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // X-axis (Columns)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // Y-axis (Rows)

    if (i <= N && j <= N) {
        int P = IDX(i, j);
        double A_E = 1.0 / (h * h); double A_W = 1.0 / (h * h);
        double A_N = 1.0 / (h * h); double A_S = 1.0 / (h * h);
        double A_P = -4.0 / (h * h);

        double du_dx = (u[IDX(i, j+1)] - u[IDX(i, j-1)]) / (2.0 * h);
        double du_dy = (u[IDX(i+1, j)] - u[IDX(i-1, j)]) / (2.0 * h);
        double dv_dx = (v[IDX(i, j+1)] - v[IDX(i, j-1)]) / (2.0 * h);
        double dv_dy = (v[IDX(i+1, j)] - v[IDX(i-1, j)]) / (2.0 * h);

        double d2u = A_E*u[IDX(i, j+1)] + A_W*u[IDX(i, j-1)] + A_N*u[IDX(i+1, j)] + A_S*u[IDX(i-1, j)] + A_P*u[P];
        double d2v = A_E*v[IDX(i, j+1)] + A_W*v[IDX(i, j-1)] + A_N*v[IDX(i+1, j)] + A_S*v[IDX(i-1, j)] + A_P*v[P];

        u_star[P] = u[P] - dt * (u[P]*du_dx + v[P]*du_dy) + dt * nu * d2u;
        v_star[P] = v[P] - dt * (u[P]*dv_dx + v[P]*dv_dy) + dt * nu * d2v;
    }
}

// DIVERGENCE KERNEL
__global__ void divergenceKernel(double* u_star, double* v_star, double* S, int N, int N_ext, double h, double dt, double rho) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N) {
        S[IDX(i, j)] = (rho / dt) * ((u_star[IDX(i, j+1)] - u_star[IDX(i, j-1)])/(2.0*h) +
                                     (v_star[IDX(i+1, j)] - v_star[IDX(i-1, j)])/(2.0*h));
    }
}

// POISSON SOLVER KERNEL
__global__ void poissonRbgsKernelFused(double* p, double* S, int N, int N_ext, double h, int num_iterations) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N) {
        int P = IDX(i, j);
        double A_E = 1.0 / (h * h); double A_W = 1.0 / (h * h);
        double A_N = 1.0 / (h * h); double A_S = 1.0 / (h * h);
        double A_P = -4.0 / (h * h);

        for (int iter = 0; iter < num_iterations; iter++) {
            // Update pressure ghosts 
            if (i == 1) {
                p[IDX(0, j)] = p[IDX(1, j)];      // Bottom
            }
            if (i == N) {
                p[IDX(N + 1, j)] = p[IDX(N, j)];  // Top
            }
            if (j == 1) {
                p[IDX(i, 0)] = p[IDX(i, 1)];      // Left
            }
            if (j == N) {
                p[IDX(i, N + 1)] = p[IDX(i, N)];  // Right
            }
            __syncthreads();

            // RED pass (color = 0)
            if ((i + j) % 2 == 0) {
                p[P] = (S[P] - (A_E*p[IDX(i, j+1)] + A_W*p[IDX(i, j-1)] + 
                                A_N*p[IDX(i+1, j)] + A_S*p[IDX(i-1, j)])) / A_P;
            }
            __syncthreads();

            // BLACK pass (color = 1)
            if ((i + j) % 2 == 1) {
                p[P] = (S[P] - (A_E*p[IDX(i, j+1)] + A_W*p[IDX(i, j-1)] + 
                                A_N*p[IDX(i+1, j)] + A_S*p[IDX(i-1, j)])) / A_P;
            }
            __syncthreads();
        }
    }
}

// CORRECTOR KERNEL
__global__ void correctorKernel(double* u, double* v, double* u_star, double* v_star, double* p, int N, int N_ext, double h, double dt, double rho) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N) {
        int P = IDX(i, j);
        
        u[P] = u_star[P] - (dt / rho) * ((p[IDX(i, j+1)] - p[IDX(i, j-1)]) / (2.0 * h));
        v[P] = v_star[P] - (dt / rho) * ((p[IDX(i+1, j)] - p[IDX(i-1, j)]) / (2.0 * h));
    }
}

// FILE I/O
void writeOutputCSV(const char* filename, double* u, double* v, double* p, int N, int N_ext, double h) {
    std::ofstream file(filename);
    file << "x,y,u,v,p\n";
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            double x = (j - 0.5) * h;
            double y = (i - 0.5) * h;
            file << x << "," << y << "," << u[IDX(i, j)] << "," << v[IDX(i, j)] << "," << p[IDX(i, j)] << "\n";
        }
    }
    file.close();
}

int main() {
    clock_t total_start = clock();

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices found: " << deviceCount << "\n";
    if (deviceCount == 0) {
        std::cerr << "No CUDA device available - aborting.\n";
        return 1;
    }

    int N = 128;
    int N_ext = N + 2;
    double h = 1.0 / N;
    double dt = 0.001;
    double nu = 0.01;
    double rho = 1.0;
    size_t total_cells = N_ext * N_ext;
    size_t bytes = total_cells * sizeof(double);

    // Host Memory
    double *h_u = new double[total_cells]();
    double *h_v = new double[total_cells]();
    double *h_p = new double[total_cells]();

    // Device Memory
    double *d_u, *d_v, *d_u_star, *d_v_star, *d_p, *d_S;
    CUDA_CHECK(cudaMalloc((void**)&d_u, bytes));      CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v, bytes));      CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_u_star, bytes)); CUDA_CHECK(cudaMemset(d_u_star, 0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v_star, bytes)); CUDA_CHECK(cudaMemset(d_v_star, 0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_p, bytes));      CUDA_CHECK(cudaMemset(d_p, 0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_S, bytes));      CUDA_CHECK(cudaMemset(d_S, 0, bytes));

    // Grid sized so N threads (indices 1..N) are always covered
    dim3 threads1D(256);
    dim3 blocks1D((N + threads1D.x - 1) / threads1D.x);
    dim3 threads2D(16, 16);
    dim3 blocks2D((N + threads2D.x - 1) / threads2D.x, (N + threads2D.y - 1) / threads2D.y);

    cudaEvent_t compute_start, compute_end;
    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_end));

    std::cout << "Starting 2D Lid-Driven Cavity Simulation (GPU - OPTIMIZED with Kernel Fusion)...\n";

    CUDA_CHECK(cudaEventRecord(compute_start));

    for (int step = 0; step < 10000; step++) {
        updateVelocityGhostsKernel<<<blocks1D, threads1D>>>(d_u, d_v, N, N_ext);

        predictorKernel<<<blocks2D, threads2D>>>(d_u, d_v, d_u_star, d_v_star, N, N_ext, h, dt, nu);

        divergenceKernel<<<blocks2D, threads2D>>>(d_u_star, d_v_star, d_S, N, N_ext, h, dt, rho);

        poissonRbgsKernelFused<<<blocks2D, threads2D>>>(d_p, d_S, N, N_ext, h, 50);

        pinPressureKernel<<<1, 1>>>(d_p, N_ext);

        correctorKernel<<<blocks2D, threads2D>>>(d_u, d_v, d_u_star, d_v_star, d_p, N, N_ext, h, dt, rho);

        // Catch a launch/execution failure as soon as it happens
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(compute_end));
    CUDA_CHECK(cudaEventSynchronize(compute_end));

    float math_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&math_time_ms, compute_start, compute_end));

    std::cout << "Data GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Writing validation data to gpu_output.csv...\n";
    writeOutputCSV("gpu_output.csv", h_u, h_v, h_p, N, N_ext, h);

    clock_t total_end = clock();

    double math_time_sec = math_time_ms / 1000.0;
    double total_time_sec = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    std::cout << "Processing Time (No I/O): " << math_time_sec << " s\n";
    std::cout << "Total Time (With I/O):    " << total_time_sec << " s\n";

    CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_u_star));
    CUDA_CHECK(cudaFree(d_v_star)); CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_S));
    delete[] h_u; delete[] h_v; delete[] h_p;
    return 0;
}
