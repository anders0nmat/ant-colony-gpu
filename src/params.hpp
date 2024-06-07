#pragma once

#include <unordered_map>
#include <string>

struct AntParams {
	double alpha;
	double beta;
	double q;
	double rho;

	double initial_pheromone;
	double min_pheromone;
	double max_pheromone;

	double zero_weight;
	uint32_t random_seed;

	std::string variant_args;
};
