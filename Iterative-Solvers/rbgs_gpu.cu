#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda_runtime.h>

/* 
 * PROBLEM SETUP: 2D Lid-Driven Cavity (GPU Navier-Stokes)
 * 
 * Physics: Simulating incompressible water inside a sealed square box 
 *          where the top wall (lid) is constantly sliding to the right.
 * Domain:  1.0m x 1.0m cavity, 128 x 128 FDM grid.
 * Fluid:   Density (rho) = 1.0, Kinematic Viscosity (nu) = 0.01 (Re = 100).
 * B.C.s:   Top Lid: u = 1.0 m/s, v = 0.0 m/s. 
 *          Sides/Bottom: u = 0.0, v = 0.0. 
 * Algorithm: Fractional Step Method (Fully Parallelized on VRAM)
*/

// KERNEL 0: Apply Moving Lid Boundary Condition
__global__ void applyBoundaryKernel(double* u, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        u[0 * N + j] = 1.0; // Top lid slides right at 1 m/s
    }
}

// KERNEL 1: Predictor Step (Solve Momentum)
__global__ void predictorKernel(double* u, double* v, double* u_star, double* v_star, 
                                int N, double h, double dt, double nu) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int P = i * N + j;
        
        double du_dx = (u[i*N + (j+1)] - u[i*N + (j-1)]) / (2.0 * h);
        double du_dy = (u[(i+1)*N + j] - u[(i-1)*N + j]) / (2.0 * h);
        double dv_dx = (v[i*N + (j+1)] - v[i*N + (j-1)]) / (2.0 * h);
        double dv_dy = (v[(i+1)*N + j] - v[(i-1)*N + j]) / (2.0 * h);
        
        double d2u = (u[(i+1)*N + j] + u[(i-1)*N + j] + u[i*N + (j+1)] + u[i*N + (j-1)] - 4.0*u[P]) / (h*h);
        double d2v = (v[(i+1)*N + j] + v[(i-1)*N + j] + v[i*N + (j+1)] + v[i*N + (j-1)] - 4.0*v[P]) / (h*h);

        u_star[P] = u[P] - dt * (u[P]*du_dx + v[P]*du_dy) + dt * nu * d2u;
        v_star[P] = v[P] - dt * (u[P]*dv_dx + v[P]*dv_dy) + dt * nu * d2v;
    }
}

// KERNEL 2: Mass Error Calculation (Source Term)
__global__ void divergenceKernel(double* u_star, double* v_star, double* S, 
                                 int N, double h, double dt, double rho) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int P = i * N + j;
        double du_star_dx = (u_star[i*N + (j+1)] - u_star[i*N + (j-1)]) / (2.0 * h);
        double dv_star_dy = (v_star[(i+1)*N + j] - v_star[(i-1)*N + j]) / (2.0 * h);
        
        S[P] = (rho / dt) * (du_star_dx + dv_star_dy);
    }
}

// KERNEL 3: Pressure Poisson Solver (Red-Black)
__global__ void poissonRbgsKernel(double* p, double* S, int N, double h, int color) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        if ((i + j) % 2 == color) {
            int P = i * N + j;
            p[P] = 0.25 * (p[(i-1)*N + j] + p[(i+1)*N + j] + p[i*N + (j+1)] + p[i*N + (j-1)] - (h*h * S[P]));
        }
    }
}

// KERNEL 4: Corrector Step (Fix Velocities)
__global__ void correctorKernel(double* u, double* v, double* u_star, double* v_star, 
                                double* p, int N, double h, double dt, double rho) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int P = i * N + j;
        double dp_dx = (p[i*N + (j+1)] - p[i*N + (j-1)]) / (2.0 * h);
        double dp_dy = (p[(i+1)*N + j] - p[(i-1)*N + j]) / (2.0 * h);
        
        u[P] = u_star[P] - (dt / rho) * dp_dx;
        v[P] = v_star[P] - (dt / rho) * dp_dy;
    }
}

int main() {
    int N = 128;                   
    double h = 1.0 / (N - 1);      
    double dt = 0.001;             
    double nu = 0.01;              
    double rho = 1.0;              
    int time_steps = 2000;         
    int poisson_iters = 50;        
    size_t bytes = N * N * sizeof(double);

    // Host Memory (RAM)
    double *h_u = new double[N * N];
    
    // Device Memory (VRAM)
    double *d_u, *d_v, *d_u_star, *d_v_star, *d_p, *d_S;
    cudaMalloc(&d_u, bytes);       cudaMemset(d_u, 0, bytes);
    cudaMalloc(&d_v, bytes);       cudaMemset(d_v, 0, bytes);
    cudaMalloc(&d_u_star, bytes);  cudaMemset(d_u_star, 0, bytes);
    cudaMalloc(&d_v_star, bytes);  cudaMemset(d_v_star, 0, bytes);
    cudaMalloc(&d_p, bytes);       cudaMemset(d_p, 0, bytes);
    cudaMalloc(&d_S, bytes);       cudaMemset(d_S, 0, bytes);

    dim3 blocks1D((N + 255) / 256);
    dim3 threads1D(256);
    dim3 blocks2D((N + 15) / 16, (N + 15) / 16);
    dim3 threads2D(16, 16);

    std::cout << "GPU: Starting 2D Lid-Driven Cavity Simulation...\n";
    clock_t start_time = clock();

    for (int step = 0; step < time_steps; step++) {
        
        applyBoundaryKernel<<<blocks1D, threads1D>>>(d_u, N);
        
        predictorKernel<<<blocks2D, threads2D>>>(d_u, d_v, d_u_star, d_v_star, N, h, dt, nu);
        
        divergenceKernel<<<blocks2D, threads2D>>>(d_u_star, d_v_star, d_S, N, h, dt, rho);
        
        for (int iter = 0; iter < poisson_iters; iter++) {
            poissonRbgsKernel<<<blocks2D, threads2D>>>(d_p, d_S, N, h, 0); // RED
            poissonRbgsKernel<<<blocks2D, threads2D>>>(d_p, d_S, N, h, 1); // BLACK
        }

        correctorKernel<<<blocks2D, threads2D>>>(d_u, d_v, d_u_star, d_v_star, d_p, N, h, dt, rho);
        
        if (step % 500 == 0) std::cout << "GPU completed time step " << step << "...\n";
    }
    
    cudaDeviceSynchronize();
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Retrieve final X-Velocity data for visualization
    cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Writing velocity data to gpu_cavity_u.txt...\n";
    std::ofstream outfile("gpu_cavity_u.txt");
    for (int i = 0; i < N * N; i++) outfile << h_u[i] << "\n";
    outfile.close();

    std::cout << "Simulation Complete!\n";
    std::cout << "Grid: " << N << "x" << N << " | Time Steps: " << time_steps << "\n";
    std::cout << "Total Math Time: " << time_spent << " seconds\n";

    delete[] h_u; 
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_u_star); 
    cudaFree(d_v_star); cudaFree(d_p); cudaFree(d_S);

    return 0;
}       