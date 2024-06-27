uint rng_minstd_rand0(uint* state) {
	const uint a = 16807;
	const uint c = 0;
	const uint m = 2147483647;

	*state = (a * (*state) + c) % m;

	return *state;
}

double rng_range(uint* state, double max) {
	uint r = rng_minstd_rand0(state);
	double dr = (double)r / (double)UINT_MAX;
	return dr * max;
}

void kernel wander_ant(
global const double* pheromone,
constant const double* visibility,
constant const int* weights,
global int* ant_routes,
global int* ant_route_length,
global double* ant_sample,
global int* ant_allowed,
int problem_size,
double alpha,
global uint* rng_seeds) {
	int ant_idx = get_global_id(0);
	int* ant_route = ant_routes + ant_idx * problem_size;
	double* sample = ant_sample + ant_idx * problem_size;
	int* allowed = ant_allowed + ant_idx * problem_size;
	uint* seed = rng_seeds + ant_idx;

	int current_node = 0;
	int* route_length = ant_route_length + ant_idx;
	*route_length = 0;
	for (int i = 1; i < problem_size; i++) {
		double sample_sum = 0.0;
		bool hasPossibleNext = false;
		for (size_t next = 0; next < problem_size; next++) {
			if (allowed[next] != 0) {
				sample[next] = 0.0;
				continue;
			}
			double edge_value = 
				powr(
					pheromone[current_node * problem_size + next],
					alpha
				) * visibility[current_node * problem_size + next];
			sample[next] = edge_value;
			sample_sum += edge_value;

			hasPossibleNext = true;
		}

		if (!hasPossibleNext) {
			current_node = -1;
			break;
		}

		double rng = rng_range(seed, sample_sum);
		double rng_before = rng;
		int next_node = -1;
		for (int i = 0; i < problem_size; i++) {
			rng -= sample[i];
			if (rng < 0) {
				next_node = i;
				break;
			}
		}
		if (next_node < 0) {
			current_node = -1;
			break;
		}

		*route_length += weights[current_node * problem_size + next_node];

		current_node = next_node;
		ant_route[i] = next_node;
		allowed[next_node] = -1;

		for (int i = 0; i < problem_size; i++) {
			if (weights[i * problem_size + next_node] == -1) {
				allowed[i] -= 1;
			}
		}
	}

	if (current_node != problem_size - 1) {
		*route_length = INT_MAX;
	}
}