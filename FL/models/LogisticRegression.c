#include <stdlib.h>

#include "mkl.h"

void 
sigmoid(int n, double *x, double *res) {
    cblas_dcopy(n, x, 1, res, 1);
    // x * (-1)
    cblas_dscal(n, -1, res, 1);
    // e ^ (-x)
    vdExp(n, res, res);
    // 1 / (1 + e^(-x))
    vdLinearFrac(n, res, res, 0, 1, 1, 1, res);
}

double
obj(int N, int dim, double *X, double *y, double *w, double reg) {
	int i;
    double *acc = (double*) malloc(sizeof(double) * N);
    double *acc2 = (double*) malloc(sizeof(double) * N);
    double c = 0.5 * reg * cblas_ddot(dim, w, 1, w, 1);
    cblas_dcopy(N, y, 1, acc, 1);
    // acc = -y
    cblas_dscal(N, -1, acc, 1);
    // acc2 = X@w
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, dim, 1, X, dim, w, 1, 0, acc2, 1);
    // acc2 = -y * (X@w)
    vdMul(N, acc, acc2, acc2);
	// acc2 = exp(acc2)
    vdExp(N, acc2, acc2);
    // acc2 += 1
    vdLinearFrac(N, acc2, acc2, 1, 1, 0, 1, acc2);
    // acc2 = log(acc2)
    vdLn(N, acc2, acc2);
    for (i = 0; i < N; ++i) {
        acc[i] = 1;
	}
    double ret = (1.0/N) * cblas_ddot(N, acc2, 1, acc, 1) + c;
    free(acc);
    free(acc2);
    return ret;
}

void
grad(int N, int dim, double *X, double *y, double *w, double reg, double* res) {
    double *acc = (double*) malloc(sizeof(double) * N);
    // Copy w into res
    cblas_dcopy(dim, w, 1, res, 1);
    // acc = X@w [acc(N x 1) = X(N x dim) @ w(dim x 1)]
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, dim, 1, X, dim, w, 1, 0, acc, 1);
    // acc = y * (X@w) 
    vdMul(N, y, acc, acc);
    // acc = sigmoid(y * (X@w)) - 1
    sigmoid(N, acc, acc);
    vdLinearFrac(N, acc, acc, 1, -1, 0, 1, acc);
    // acc = y * (sigmoid(...) - 1)
    vdMul(N, y, acc, acc);
    // 1/N * X.T@(y * (sigmoid(...) - 1)+ reg*w
    cblas_dgemv(CblasRowMajor, CblasTrans, N, dim, 1.0/N, X, dim, acc, 1, reg, res, 1);
    free(acc);
}
