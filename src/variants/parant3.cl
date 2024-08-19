
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
global double* ant_sample,
global int* ant_allowed,
int problem_size,
global uint* rng_seeds) {
	int ant_idx = get_group_id(1);
	int worker_idx = get_local_id(0);

	int* ant_route = ant_routes + ant_idx * problem_size;
	double* sample = ant_sample + ant_idx * problem_size;
	int* allowed = ant_allowed + ant_idx * problem_size;
	uint* seed = rng_seeds + ant_idx;
	const int bitmask_size = problem_size / BITMASK_SIZE + (problem_size % BITMASK_SIZE != 0 ? 1 : 0);

	double* worker_sample = sample + worker_idx;
	int* worker_allowed = allowed + worker_idx;

	local bool stuck_flag;
	local int current_node;
	local double rng;
	local int next_node;
	if (worker_idx == 0) {
		stuck_flag = false;
		current_node = 0;
		rng = 0.0;
		next_node = -1;
	}
	int* route_length = ant_route_length + ant_idx;
	*route_length = 0;
	for (int i = 1; i < problem_size; i++) {
		barrier(CLK_LOCAL_MEM_FENCE);
		sample[worker_idx] = allowed[worker_idx] == 0 ? probabilities[current_node * problem_size + worker_idx] : 0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (worker_idx == 0) {
			double sample_sum = 0.0;
			// Prefix Sum
			for (int ii = 0; ii < problem_size; ii++) {
				sample_sum += sample[ii];
				sample[ii] = sample_sum;
			}

			rng = rng_range(seed, sample_sum);
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
				*route_length += weights[current_node * problem_size + next_node];

				current_node = next_node;
				ant_route[i] = next_node;
				allowed[next_node] = -1;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		const bitmask* dep_mask = dependencies + current_node * bitmask_size;
		if (!stuck_flag && has_bit(dep_mask, worker_idx)) {
			// node `worker_idx` depends on current_node => Update allowed entry
			allowed[worker_idx] -= 1;
		}
	}

	if (worker_idx == 0 && current_node != problem_size - 1) {
		*route_length = INT_MAX;
	}
}


void kernel update_pheromone(
global double* pheromone,
global double* probabilities,
global double* visibility,
double alpha,
double one_minus_roh,
double min_pheromone,
double max_pheromone,
global const int* ant_routes,
uint best_ant_idx,
double best_ant_pheromone,
int problem_size
) {
	int edge = get_global_id(0);
	int from = edge / problem_size;
	int to = edge % problem_size;
	const int* best_ant_route = ant_routes + best_ant_idx * problem_size;

	// Evaporate
	pheromone[edge] *= one_minus_roh;

	// Lay along best_ant
	//? Could be "optimized" by looping one short, thereby removing the check for i+1 to be a valid index

	//? Could also be made obsolete if a bitmask exists, containing each edge and whether it is part of the best_route (1) or not (0)

	//? How about a list of each node containing the next node in the route
	//? This would require reimagining the meaning of ant_route but nothing more
	//? Because each node MUST occure in the route, there are as many entries as there are nodes
	//? Currently, ant_route[index] is the node where ant was at step no #index
	//? But ant_route[index] could also be interpreted as the next node that was chosen while standing at node "index"
	//? No information would be lost (although retrieving the step-by-step best route would be slightly more complex)
	//? But the following algorithm would be made possible
	/*
	? next_arr[problem_size]
	? if next_arr[from] == to {
	? 	pheromone[edge] += best_ant_pheromone
	? }
	*/
	for (int i = 0; i < problem_size; i++) {
		if (best_ant_route[i] != from) {
			continue;
		}

		if (i + 1 >= problem_size) {
			 continue;
		}

		if (best_ant_route[i + 1] != to) {
			continue;
		}

		pheromone[edge] += best_ant_pheromone;
	}

	pheromone[edge] = clamp(pheromone[edge], min_pheromone, max_pheromone);

	probabilities[edge] = powr(pheromone[edge], alpha) * visibility[edge];
}

void kernel reset_allowed(
global const int* allowed_template,
global int* allowed_data
) {
	int id = get_global_id(0);
	allowed_data[id] = allowed_template[id];
}

