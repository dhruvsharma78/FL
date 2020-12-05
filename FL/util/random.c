#include "random.h"
#include "mkl.h"

void
init_rng(VSLStreamStatePtr *stream, long long *_seed) {
	long long seed;
	if (_seed != NULL) {
		seed = *_seed;
	} else {
		seed = time(NULL) * 256;
	}
	vslNewStream(stream, VSL_BRNG_MT19937, seed);
}

void
fill_random_ints (VSLStreamStatePtr stream, int n, int *r, int a, int b) {
	int err;
	err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, a, b);
}

void
fill_random_doubles (VSLStreamStatePtr stream, int n, double *r) {
	int err;
	err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n, r, 0, 1);
}
