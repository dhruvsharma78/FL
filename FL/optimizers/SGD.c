#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "omp.h"
#include "mkl.h"
#include "SGD.h"
#include "../util/random.h"

void
fill_X_batch (double *X, int batch_size, int dim, double *X_b, int *rand_idx) {
	int i, j;
	for (i = 0; i < batch_size; ++i) {
		for (j = 0; j < dim; ++j) {
			X_b[i * dim + j] = X[rand_idx[i] * dim + j];
		}
	}
}

void
fill_y_batch (double *y, int batch_size, double *y_b, int *rand_idx) {
	int i;
	for (i = 0; i < batch_size; ++i) {
		y_b[i] = y[rand_idx[i]];
	}
}

void
sgd_task(sgd_context_t* ctx, int start, double *_obj_SGD, double *_obj_SGD_iters, double *_MSE, double *_w) {
	int i, j, k, max_iters;
	int *rand_idx;
	double step;
	double *obj_SGD, *obj_SGD_iters, *MSE, *w, *X_batch, *y_batch, *temp;
	VSLStreamStatePtr rngstream;
	init_rng(&rngstream, NULL);
	max_iters = ctx->N / ctx->batch_size;
	obj_SGD_iters = &_obj_SGD_iters[start * ctx->epochs * max_iters];
	obj_SGD       = &_obj_SGD[start * ctx->epochs];
	MSE           = &_MSE[start * ctx-> epochs * max_iters];
	w             = &_w[start * ctx->dim];
	X_batch       = (double*) malloc(sizeof(double) * ctx->batch_size * ctx->dim);
	y_batch       = (double*) malloc(sizeof(double) * ctx->batch_size);
	temp          = (double*) malloc(sizeof(double) * ctx->dim);
	rand_idx      = (int*)    malloc(sizeof(int) * ctx->batch_size);
	if (ctx->w_next == NULL) {
		fill_random_doubles(rngstream, ctx->dim, w);
		cblas_dscal(ctx->dim, 0.001, w, 1);
	} else {
		cblas_dcopy(ctx->dim, ctx->w_next, 1, w, 1);
	}
	for (j = 0; j < ctx->epochs; ++j) {
		obj_SGD[j] += ctx->objective_func(ctx->N, ctx->dim, ctx->X, ctx->y, w, ctx->reg);
		if (!(j % 10)) {
			printf("Experiment: %d/%d, Epoch %d/%d, Loss: %f\n", start+1, ctx->experiments, j+1, ctx->epochs, obj_SGD[j]);
		}
		for (i = 0; i < max_iters; ++i) {
			fill_random_ints(rngstream, ctx->batch_size, rand_idx, 0, ctx->N - 1);
			fill_X_batch(ctx->X, ctx->batch_size, ctx->dim, X_batch, rand_idx);
			fill_y_batch(ctx->y, ctx->batch_size, y_batch, rand_idx);
			obj_SGD_iters[j * max_iters + i] += ctx->objective_func(ctx->N, ctx->dim, ctx->X, ctx->y, w, ctx->reg);
			// temp = w - w_star
			vdSub(ctx->dim, w, ctx->w_star, temp);
			MSE[j * max_iters + i] += cblas_dnrm2(ctx->dim, temp, 1); // Euclidian norm
			step = ctx->decay == 0 ? ctx->step_size : 0.1 / (ctx->reg * (j+1000));
			// temp = gradient vector
			ctx->gradient_func(ctx->batch_size, ctx->dim, X_batch, y_batch, w, ctx->reg, temp);
			// Multiply gradient vector by scalar `temp`
			cblas_dscal(ctx->dim, step, temp, 1);
			// Subtract scaled gradient from w
			vdSub(ctx->dim, w, temp, w);
		}
	}
	free(X_batch);
	free(y_batch);
	free(temp);
	free(rand_idx);
}

result_t
run (sgd_context_t *ctx) {

	int max_iters, i, j, k;
	double *_obj_SGD, *_obj_SGD_iters, *_MSE, *_w;
	double *obj_SGD, *obj_SGD_iters, *MSE, *w;
	result_t result;

	max_iters = ctx->N / ctx->batch_size;

	_obj_SGD       = (double*) calloc(ctx->experiments * ctx->epochs            , sizeof(double));
	_obj_SGD_iters = (double*) calloc(ctx->experiments * ctx->epochs * max_iters, sizeof(double));
	_MSE           = (double*) calloc(ctx->experiments * ctx->epochs * max_iters, sizeof(double));
	_w             = (double*) calloc(ctx->experiments * ctx->dim               , sizeof(double));

	if (ctx->parallel == 1) {
#pragma omp parallel
		{
#pragma omp single
			{
				for (i = 0; i < ctx->experiments; ++i) {
#pragma omp task firstprivate(i)
					sgd_task(ctx, i, _obj_SGD, _obj_SGD_iters, _MSE, _w);
				}
			}
		}
	} else {
		for (i = 0; i < ctx->experiments; ++i) {
			sgd_task(ctx, i, _obj_SGD, _obj_SGD_iters, _MSE, _w);
		}
	}

	obj_SGD       = (double*) calloc(ctx->epochs, sizeof(double));
	obj_SGD_iters = (double*) calloc(ctx->epochs * max_iters, sizeof(double));
	MSE           = (double*) calloc(ctx->epochs * max_iters, sizeof(double));
	w             = (double*) calloc(ctx->dim, sizeof(double));

	for (i = 0; i < ctx->experiments; ++i) {
		for (j = 0; j < ctx->dim; ++j) {
			w[j] += _w[i * ctx->dim + j] / ctx->experiments;
		}
		for (j = 0; j < ctx->epochs; ++j) {
            obj_SGD[j] += _obj_SGD[i * ctx->epochs + j] / ctx->experiments;
			for (k = 0; k < max_iters; ++k) {
				obj_SGD_iters[k] += _obj_SGD_iters[i * (max_iters * ctx->epochs) + j * (max_iters) + k] / ctx->experiments;
				MSE[k]           += _obj_SGD_iters[i * (max_iters * ctx->epochs) + j * (max_iters) + k] / ctx->experiments;
			}
		}
	}

	result.obj_SGD       = obj_SGD;
	result.obj_SGD_iters = obj_SGD_iters;
	result.MSE           = MSE;
	result.w             = w;


	free(_MSE);
	free(_obj_SGD);
	free(_obj_SGD_iters);
	free(_w);

	return result;

}

result_t
run_local (sgd_context_t *ctx) {

	int exp, experiments, worker, workers, j, num_split, comm_round;
	double *aggregate_w, *temp, *obj_SGD, *MSE_SGD, **worker_results;
	result_t res;

	workers = ctx->workers;
	aggregate_w = (double*) calloc(ctx->dim, sizeof(double));
	temp = (double*) malloc(ctx->dim * sizeof(double));
	MSE_SGD = (double*) calloc(ctx->communications, sizeof(double));
	obj_SGD = (double*) calloc(ctx->communications, sizeof(double));
	worker_results = (double**) malloc(sizeof(double*) * workers);
	experiments = ctx->experiments;
	num_split = ctx->N / workers;

#pragma omp parallel
#pragma omp single
	{
		for (exp = 0; exp < experiments; ++exp) {
			for (comm_round = 0; comm_round < ctx->communications; ++comm_round) {
				for (worker = 0; worker < workers; ++worker) {
#pragma omp task firstprivate(worker)
					{
						sgd_context_t subtask_ctx;
						result_t subtask_result;
						memcpy(&subtask_ctx, ctx, sizeof(sgd_context_t));
						subtask_ctx.parallel = 0;
						subtask_ctx.N = num_split;
						subtask_ctx.X = &subtask_ctx.X[worker * num_split * ctx->dim];
						subtask_ctx.y = &subtask_ctx.y[worker * num_split];
						subtask_ctx.experiments = 1;
						subtask_result = run(&subtask_ctx);
						worker_results[worker] = subtask_result.w;
						/*printf("%d", worker*num_split)*/
					}
				}
#pragma omp taskwait
				for (j = 0; j < ctx->dim; ++j) {
					aggregate_w[j] = 0;
				}
				for (j = 0; j < workers; ++j) {

					vdAdd(ctx->dim, aggregate_w, worker_results[j], aggregate_w);
				}
				cblas_dscal(ctx->dim, (1.0 / workers), aggregate_w, 1);
				ctx->w_next = aggregate_w;
				obj_SGD[comm_round] += ctx->objective_func(ctx->N, ctx->dim, ctx->X, ctx->y, aggregate_w, ctx->reg);
				vdSub(ctx->dim, aggregate_w, ctx->w_star, temp);
				MSE_SGD[comm_round] += cblas_dnrm2(ctx->dim, temp, 1);
				printf("LocalSGD: Master Thread: Experiment %d/%d, Communication %d/%d, Loss %f\n\n",
						exp + 1, experiments, comm_round + 1, ctx->communications, obj_SGD[comm_round]/(exp+1));
			}
		}
	}

	// obj_SGD / experiments
	cblas_dscal(ctx->communications, (1.0 / experiments), obj_SGD, 1);
	// MSE / experiments
	cblas_dscal(ctx->communications, (1.0 / experiments), MSE_SGD, 1);
	// log( ... )
	vdLn(ctx->communications, MSE_SGD, MSE_SGD);
	cblas_dscal(ctx->communications, 10, MSE_SGD, 1);

	res.w = aggregate_w;
	res.obj_SGD = obj_SGD;
	res.MSE = MSE_SGD;

	free(temp);

	return res;

}
