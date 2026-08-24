#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>

/* 
 * PROBLEM SETUP: 2D Lid-Driven Cavity (Navier-Stokes)
 * 
 * Physics: Simulating incompressible water inside a sealed square box 
 *          where the top wall (lid) is constantly sliding to the right.
 *          This creates a massive central vortex.
 * Domain:  1.0m x 1.0m cavity, 128 x 128 FDM grid.
 * Fluid:   Density (rho) = 1.0, Kinematic Viscosity (nu) = 0.01 (Re = 100).
 * B.C.s:   Top Lid: u = 1.0 m/s, v = 0.0 m/s. (Moving wall)
 *          Sides/Bottom: u = 0.0, v = 0.0. (No-slip stationary walls)
 *          Pressure: Neumann at walls, solved via Red-Black GS.
 * Algorithm: Fractional Step Method (Predictor -> Poisson -> Corrector)
*/

int main() {
    // 1. Grid & Fluid Properties
    int N = 128;                   // Grid resolution (scaled down for CPU speed)
    double h = 1.0 / (N - 1);      // Spatial step size (dx = dy = h)
    double dt = 0.001;             // Time step size
    double nu = 0.01;              // Kinematic viscosity
    double rho = 1.0;              // Density
    int time_steps = 2000;         // Total time loops to run
    int poisson_iters = 50;        // Red-Black sweeps per time step
    int total_cells = N * N;

    // 2. Raw Memory Allocation (Host)
    double* u = new double[total_cells];      // X-Velocity
    double* v = new double[total_cells];      // Y-Velocity
    double* u_star = new double[total_cells]; // Guessed X-Velocity
    double* v_star = new double[total_cells]; // Guessed Y-Velocity
    double* p = new double[total_cells];      // Pressure
    double* S = new double[total_cells];      // Mass error (Source)

    // Initialize all to zero (stagnant water)
    for (int i = 0; i < total_cells; i++) {
        u[i] = 0.0; v[i] = 0.0;
        u_star[i] = 0.0; v_star[i] = 0.0;
        p[i] = 0.0; S[i] = 0.0;
    }

    std::cout << "Starting 2D Lid-Driven Cavity Simulation...\n";
    clock_t start_time = clock();

    for (int step = 0; step < time_steps; step++) {
        
        // APPLY BOUNDARY CONDITIONS (The Lid)
        for (int j = 0; j < N; j++) {
            u[0 * N + j] = 1.0; // Top row sliding right at 1 m/s
        }

        // STEP 1: VELOCITY PREDICTOR (Solve Momentum Equation)
        // Guess where the fluid wants to go based on its current momentum and viscosity
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int P = i * N + j;
                
                // FDM Central Differences for gradients
                double du_dx = (u[i * N + (j + 1)] - u[i * N + (j - 1)]) / (2.0 * h);
                double du_dy = (u[(i + 1) * N + j] - u[(i - 1) * N + j]) / (2.0 * h);
                double dv_dx = (v[i * N + (j + 1)] - v[i * N + (j - 1)]) / (2.0 * h);
                double dv_dy = (v[(i + 1) * N + j] - v[(i - 1) * N + j]) / (2.0 * h);
                
                double d2u = (u[(i + 1)*N + j] + u[(i - 1)*N + j] + u[i*N + (j + 1)] + u[i*N + (j - 1)] - 4.0*u[P]) / (h * h);
                double d2v = (v[(i + 1)*N + j] + v[(i - 1)*N + j] + v[i*N + (j + 1)] + v[i*N + (j - 1)] - 4.0*v[P]) / (h * h);

                // Temporary guessed velocities (Advection + Diffusion)
                u_star[P] = u[P] - dt * (u[P] * du_dx + v[P] * du_dy) + dt * nu * d2u;
                v_star[P] = v[P] - dt * (u[P] * dv_dx + v[P] * dv_dy) + dt * nu * d2v;
            }
        }

        // STEP 2: MEASURE MASS VIOLATION (The Source Term)
        // Calculate the divergence of the guessed velocities
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int P = i * N + j;
                double du_star_dx = (u_star[i * N + (j + 1)] - u_star[i * N + (j - 1)]) / (2.0 * h);
                double dv_star_dy = (v_star[(i + 1) * N + j] - v_star[(i - 1) * N + j]) / (2.0 * h);
                
                S[P] = (rho / dt) * (du_star_dx + dv_star_dy);
            }
        }

        // STEP 3: PRESSURE POISSON SOLVER (The Red-Black GS Engine)
        // Find the pressure required to fix the mass violation
        for (int iter = 0; iter < poisson_iters; iter++) {
            // RED Sweep
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    if ((i + j) % 2 == 0) {
                        int P = i * N + j;
                        p[P] = 0.25 * (p[(i - 1)*N + j] + p[(i + 1)*N + j] + p[i*N + (j + 1)] + p[i*N + (j - 1)] - (h * h * S[P]));
                    }
                }
            }
            // BLACK Sweep
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    if ((i + j) % 2 == 1) {
                        int P = i * N + j;
                        p[P] = 0.25 * (p[(i - 1)*N + j] + p[(i + 1)*N + j] + p[i*N + (j + 1)] + p[i*N + (j - 1)] - (h * h * S[P]));
                    }
                }
            }
        }

        // STEP 4: VELOCITY CORRECTOR 
        // Use the new pressure field to force water to obey conservation of mass
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int P = i * N + j;
                double dp_dx = (p[i * N + (j + 1)] - p[i * N + (j - 1)]) / (2.0 * h);
                double dp_dy = (p[(i + 1) * N + j] - p[(i - 1) * N + j]) / (2.0 * h);
                
                u[P] = u_star[P] - (dt / rho) * dp_dx;
                v[P] = v_star[P] - (dt / rho) * dp_dy;
            }
        }

        // Progress tracker
        if (step % 500 == 0) std::cout << "Completed time step " << step << "...\n";
    }

    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Writing velocity data to cpu_cavity_u.txt...\n";
    std::ofstream outfile("cpu_cavity_u.txt");
    for (int i = 0; i < total_cells; i++) {
        outfile << u[i] << "\n";
    }
    outfile.close();

    std::cout << "Simulation Complete!\n";
    std::cout << "Grid: " << N << "x" << N << " | Time Steps: " << time_steps << "\n";
    std::cout << "Total Math Time: " << time_spent << " seconds\n";

    delete[] u; delete[] v;
    delete[] u_star; delete[] v_star;
    delete[] p; delete[] S;

    return 0;
}