#ifndef __FL_UTIL_RANDOM_H
#define __FL_UTIL_RANDOM_H

#include "mkl.h"

void init_rng(VSLStreamStatePtr *, long long *);
void fill_random_ints (VSLStreamStatePtr, int, int *, int, int);
void fill_random_doubles (VSLStreamStatePtr, int, double *);

#endif
