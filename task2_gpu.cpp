#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;

#define size 512
#define tol 0.000001
#define iter_max 100

int main(int argc, char* argv[]) {
    auto begin = std::chrono::steady_clock::now();

    //double tol = atof(argv[1]);
    //int size = atoi(argv[2]), iter_max = atoi(argv[3]);

    double** A = new double*[size];
    for (size_t i = 0; i < size; ++i) A[i] = new double[size];
    double** Anew = new double*[size];
    for (size_t i = 0; i < size; ++i) Anew[i] = new double[size];

    int iter = 0;
    double error = 1.0;
    double add = 10.0 / (size - 1);

    A[0][0] = 10;
    A[0][size - 1] = 20;
    A[size - 1][0] = 20;
    A[size - 1][size - 1] = 30;

    #pragma acc kernels
    for (int i = 1; i < size - 1; i++) {
        A[0][i] = A[0][i - 1] + add;
        A[size - 1][i] = A[size - 1][i - 1] + add;
        A[i][0] = A[i - 1][0] + add;
        A[i][size - 1] = A[i - 1][size - 1] + add;
    }

    #pragma acc data copyin(A[0:size][0:size], Anew[0:size][0:size])
    {
    while ((error > tol) && (iter < iter_max)) {
        iter = iter + 1;
        error = 0.0;

        #pragma acc kernels
        {
        for (int j = 1; j < size - 1; j++) {
            for (int i = 1; i < size - 1; i++) {
                Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                error = fmax(error, fabs(Anew[i][j] - A[j][i]));
            }
        }
        for ( int i = 1; i < size - 1; i++)
        {
            for( int j = 1; j < size - 1; j++ )
            {
                A[i][j] = Anew[i][j];
            }
        }
        }
        if ((iter % 100 == 0) or (iter == 1)) {
            std::cout << iter << ":" << error << "\n";
        }
    }
    }

    for (size_t i = 0; i < size; ++i) delete[] A[i];
    delete[] A;
    for (size_t i = 0; i < size; ++i) delete[] Anew[i];
    delete[] Anew;

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}

