#include <stdio.h>
#include <time.h>

#define N 25

int main() {
    float a[N] = {
         1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    float b[N] = {
        25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10,  9,  8,  7,  6,
         5,  4,  3,  2,  1
    };
    float c[N];

    clock_t t0 = clock();
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    clock_t t1 = clock();

    double elapsed_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
    printf("Serial CPU addition time: %.3f ms\n\n", elapsed_ms);

    for (int i = 0; i < N; i++) {
        printf("%.0f + %.0f = %.0f\n", a[i], b[i], c[i]);
    }

    return 0;
}
