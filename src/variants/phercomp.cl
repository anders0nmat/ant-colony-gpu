
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
global const double* probabilities,
global const int* weights,
global int* ant_routes,
global int* ant_route_length,
global double* ant_sample,
global int* ant_allowed,
int problem_size,
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
		/*
		? Two bitsets:
		? 	visited[node] => Use the inverse (has_to_be_visited) instead, better for later calculation
		? 	dependencies[node][node]
		? Then, Update would be obsolete, because only update needed is visited[next_node] = true
		? Check for allowed[next_node] would be
		? 	!visited[next_node] for already visited nodes
		? 	OR has_to_be_visited[next_node]
		? 	dependencies[next_node] & ~visited == 0 for satisfying all dependencies
		? 	dependent: 1001
		? 	visited:   1000
		? 	~visited:  0111 = 0001 != 0
		? 	visited:   1100
		? 	~visited:  0011 = 0001 != 0
		? 	visited:   1101
		? 	~visited:  0010 = 0000 == 0
		*/
		for (size_t next = 0; next < problem_size; next++) {
			if (allowed[next] != 0) {
				sample[next] = 0.0;
				continue;
			}
			double edge_value = probabilities[current_node * problem_size + next];
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
		//? How about a binary-search-esque sample approach
		//? This would need running-total probabilities, e.g. [2.1, 2.5, 2.5, 3.4, 5.2]
		//? Beware thath there are duplicate probabilities, in this case, the first one should be chosen
		//? This is because impossible choices are inserted with a running total of total+0.0, resulting in a duplicate entry
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

		//? Bitmask representing dependent nodes
		/*
		? uint dependent_bitmasks[];
		? uint length = length(dependent_bitmask);
		? uint bitidx = 0;
		? uint bitmask = dependent[bitindex];
		? while true {
		? 	uint first_nonzero = clz(bitmask);
		? 	if first_nonzero == sizeof(uint)*8 {
		? 		bitidx++;
		? 		if bitidx >= length { break }
		? 		bitmask = dependent[bitidx]
		? 	}
		? 	else {
		? 		allowed[first_nonzero + sizeof(uint)*8*bitidx] -= 1;
		? 		bitmask &= ~(1 << (sizeof(uint)*8 - first_nonzero));
		? 	}
		? }
		*/

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

