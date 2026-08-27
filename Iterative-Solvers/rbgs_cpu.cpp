#include <iostream>
#include <fstream>
#include <ctime>

/*
SETUP: 2D Lid-Driven Cavity 
*/

// Macro to convert 2D coordinates to 1D flat array index
#define IDX(i, j) ((i) * N_ext + (j))

void updateVelocityGhosts(double* u, double* v, int N, int N_ext) {
    for (int k = 1; k <= N; k++) {
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

void updatePressureGhosts(double* p, int N, int N_ext) {
    for (int k = 1; k <= N; k++) {
        p[IDX(N + 1, k)] = p[IDX(N, k)]; // Top (Row N+1)
        p[IDX(0, k)]     = p[IDX(1, k)]; // Bottom (Row 0)
        p[IDX(k, 0)]     = p[IDX(k, 1)]; // Left (Col 0)
        p[IDX(k, N + 1)] = p[IDX(k, N)]; // Right (Col N+1)
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

    int N = 256;                   
    int N_ext = N + 2;             
    double h = 1.0 / N;            
    double dt = 0.0003;             
    double nu = 0.01;              
    double rho = 1.0;              
    int total_cells = N_ext * N_ext;

    double* u = new double[total_cells]();
    double* v = new double[total_cells]();
    double* u_star = new double[total_cells]();
    double* v_star = new double[total_cells]();
    double* p = new double[total_cells]();
    double* S = new double[total_cells]();

    double A_E = 1.0 / (h * h); double A_W = 1.0 / (h * h);
    double A_N = 1.0 / (h * h); double A_S = 1.0 / (h * h);
    double A_P = -4.0 / (h * h);

    std::cout << "Starting 2D Lid-Driven Cavity Simulation...\n";
    
    clock_t math_start = clock();

    // MAIN LOOP
    for (int step = 0; step < 10000; step++) {
        updateVelocityGhosts(u, v, N, N_ext);

        // THE PREDICTOR
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int P = IDX(i, j);
                
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

        // DIVERGENCE
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                // Corrected: 'u' differentiates across 'j', 'v' differentiates across 'i'
                S[IDX(i, j)] = (rho / dt) * ((u_star[IDX(i, j+1)] - u_star[IDX(i, j-1)])/(2.0*h) + 
                                             (v_star[IDX(i+1, j)] - v_star[IDX(i-1, j)])/(2.0*h));
            }
        }

        // THE POISSON SOLVER
        for (int iter = 0; iter < 50; iter++) {
            updatePressureGhosts(p, N, N_ext); 
            for (int color = 0; color < 2; color++) { 
                for (int i = 1; i <= N; i++) {
                    for (int j = 1; j <= N; j++) {
                        if ((i + j) % 2 == color) {
                            int P = IDX(i, j);
                            // Corrected neighbor mapping
                            p[P] = (S[P] - (A_E*p[IDX(i, j+1)] + A_W*p[IDX(i, j-1)] + 
                                            A_N*p[IDX(i+1, j)] + A_S*p[IDX(i-1, j)])) / A_P;
                        }
                    }
                }
            }
        }
        p[IDX(1, 1)] = 0.0;

        // THE CORRECTOR
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int P = IDX(i, j);
                // Corrected: Pressure gradient pushing 'u' is dp/dx (changes 'j')
                u[P] = u_star[P] - (dt / rho) * ((p[IDX(i, j+1)] - p[IDX(i, j-1)]) / (2.0 * h));
                // Corrected: Pressure gradient pushing 'v' is dp/dy (changes 'i')
                v[P] = v_star[P] - (dt / rho) * ((p[IDX(i+1, j)] - p[IDX(i-1, j)]) / (2.0 * h));
            }
        }
    }

    clock_t math_end = clock(); 
    
    std::cout << "Writing validation data to cpu_output.csv...\n";
    writeOutputCSV("cpu_output.csv", u, v, p, N, N_ext, h);
    
    clock_t total_end = clock(); 

    // Calculate times in seconds
    double math_time = (double)(math_end - math_start) / CLOCKS_PER_SEC;
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    std::cout << "Processing Time (No I/O): " << math_time << " s\n";
    std::cout << "Total Time (With I/O):    " << total_time << " s\n";
    
    delete[] u; delete[] v; delete[] u_star; 
    delete[] v_star; delete[] p; delete[] S;
    return 0;
}