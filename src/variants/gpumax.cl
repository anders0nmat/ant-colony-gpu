
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

#if defined(__opencl_c_int64) && !defined(FORCE_32BITMASK)
typedef ulong bitmask;
const uint BITMASK_SIZE = 64;
#else
typedef uint bitmask;
const uint BITMASK_SIZE = 32;
#endif

inline void reset_bit(bitmask* mask, uint index) {
	uint mask_idx = index / BITMASK_SIZE;
	uint bit_idx = index % BITMASK_SIZE;

	mask[mask_idx] &= ~(1UL << bit_idx);
}

inline void set_bit(bitmask* mask, uint index) {
	uint mask_idx = index / BITMASK_SIZE;
	uint bit_idx = index % BITMASK_SIZE;

	mask[mask_idx] |= (1UL << bit_idx);
}

inline bool has_bit(const bitmask* mask, uint index) {
	uint mask_idx = index / BITMASK_SIZE;
	uint bit_idx = index % BITMASK_SIZE;

	return (mask[mask_idx] & (1UL << bit_idx)) != 0;
}

void kernel wander_ant(
global const double* probabilities,
global const int* weights,
global const bitmask* dependencies,
global int* ant_routes,
global int* ant_route_length,
local double* ant_sample,
local int* ant_allowed,
global const int* ant_allowed_template,
int problem_size,
global uint* rng_seeds) {
	int ant_idx = get_group_id(1);
	int worker_idx = get_local_id(0);
	int worker_size = get_local_size(0);
	int clipped_idx = min(worker_idx, problem_size - 1);

	int* ant_route = ant_routes + ant_idx * problem_size;
	double* sample = ant_sample;
	int* allowed = ant_allowed;
	uint* seed = rng_seeds + ant_idx;
	const int bitmask_size = problem_size / BITMASK_SIZE + (problem_size % BITMASK_SIZE != 0 ? 1 : 0);

	double* worker_sample = sample + worker_idx;
	int* worker_allowed = allowed + worker_idx;
	int route_length = 0;

	const int worker_pwr = ctz(~worker_idx);
	const int max_pwr = ctz(worker_size) + 1;
	bool stuck_flag = worker_idx >= problem_size;
	local int current_node;
	local double rng;
	local int next_node;
	if (worker_idx == 0) {
		current_node = 0;
		rng = 0.0;
		next_node = -1;
	}
	if (worker_idx < problem_size) {
		allowed[worker_idx] = ant_allowed_template[worker_idx];
	}
	for (int i = 1; i < problem_size; i++) {
		barrier(CLK_LOCAL_MEM_FENCE);
		sample[worker_idx] = allowed[clipped_idx] == 0 ? probabilities[current_node * problem_size + clipped_idx] : 0;

		// Up-Sweep
		double my_sample = sample[worker_idx]; // Added at the end
		for (int i = 0; i < max_pwr; i++) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (worker_pwr > i) {
				int merge_idx = 1UL << i;
				sample[worker_idx] += sample[worker_idx - merge_idx];
			}
		}

		if (worker_idx == worker_size - 1) {
			sample[worker_idx] = 0;
		}

		// Down-Sweep
		for (int i = max_pwr - 1; i >= 0; i--) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (worker_pwr > i) {
				int merge_idx = 1UL << i;
				double curr = sample[worker_idx];
				sample[worker_idx] += sample[worker_idx - merge_idx];
				sample[worker_idx - merge_idx] = curr;	
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		sample[worker_idx] += my_sample; // Make exclusive scan into inclusive one
		
		if (worker_idx == problem_size - 1) {
			rng = rng_range(seed, sample[worker_idx]);
			next_node = -1;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		bool in_self_range = 
			rng < sample[worker_idx]
			&& (worker_idx == 0 || rng >= sample[worker_idx - 1]);
		if (in_self_range) {
			next_node = worker_idx;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		if (worker_idx == 0) {
			if (next_node < 0) {
				stuck_flag = true;
			}

			if (!stuck_flag) {
				route_length += weights[current_node * problem_size + next_node];

				ant_route[current_node] = next_node;
				allowed[next_node] = -1;
				current_node = next_node;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		const bitmask* dep_mask = dependencies + current_node * bitmask_size;
		if (!stuck_flag && has_bit(dep_mask, clipped_idx)) {
			// node `worker_idx` depends on current_node => Update allowed entry
			allowed[clipped_idx] -= 1;
		}
	}

	if (worker_idx == 0) {
		ant_route_length[ant_idx] = (current_node == problem_size - 1) ? route_length : INT_MAX;
	}
}

global int best_ant_idx = 0;
global double best_ant_pheromone = 0.0;

void kernel update_pheromone(
global double* pheromone,
global double* probabilities,
global double* visibility,
double alpha,
double one_minus_roh,
double min_pheromone,
double max_pheromone,
global const int* ant_routes,
int problem_size
) {
	int edge = get_global_id(0);
	int from = edge / problem_size;
	int to = edge % problem_size;
	const int* best_ant_route = ant_routes + best_ant_idx * problem_size;

	// Evaporate
	pheromone[edge] *= one_minus_roh;

	// Lay along best_ant
	if (best_ant_route[from] == to) {
		pheromone[edge] += best_ant_pheromone;
	}

	pheromone[edge] = clamp(pheromone[edge], min_pheromone, max_pheromone);

	probabilities[edge] = powr(pheromone[edge], alpha) * visibility[edge];
}

void kernel get_best_ant(
global int* ant_route_length,
global int* best_length,
double pheromone,
int problem_size
) {
	int best_len = INT_MAX;
	int best_idx = -1;
	for (int i = 0; i < problem_size; i++) {
		if (best_len > ant_route_length[i]) {
			best_len = ant_route_length[i];
			best_idx = i;
		}
	}

	best_ant_idx = best_idx;
	best_ant_pheromone = pheromone / best_len;
	*best_length = min(*best_length, best_len);
}
