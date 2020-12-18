#ifndef __FL_SGD_H
#define __FL_SGD_H

typedef struct sgd_context_t {
	double* X;
	double* y;
	double* w_next;
	double* w_star;
	double reg;
	double step_size;
	int N;
	int dim;
	int decay;
	int epochs;
	int workers;
	int parallel;
	int batch_size;
	int print_updates;
	int experiments;
	int communications;
	double (*objective_func) (int N, int dim, double*, double*, double*, double);
	void   (*gradient_func)  (int N, int dim, double*, double*, double*, double, double*);
} sgd_context_t;

typedef struct result_t {
	double *obj_SGD;
	double *obj_SGD_iters;
	double *MSE;
	double *w;
} result_t;

result_t run (sgd_context_t *);
result_t run_local (sgd_context_t *);
void sgd_task(sgd_context_t *, int, double *, double *, double *, double *);
void fill_y_batch (double *y, int batch_size, double *y_b, int *rand_idx);
void fill_X_batch (double *X, int batch_size, int dim, double *X_b, int *rand_idx);

#endif
