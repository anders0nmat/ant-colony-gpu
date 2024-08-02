#pragma once

#include <algorithm>
#include <random>

#include "clcolony.hpp"
#include "../profiler.hpp"

class ManyAntOptimizer: public CLColonyOptimizer {
protected:
	cl::Program program;
	cl::KernelFunctor<
		cl::Buffer, // pheromone
		cl::Buffer, // visibility
		cl::Buffer, // weights
		cl::Buffer, // ant_routes
		cl::Buffer, // ant_routes_length
		cl::Buffer, // ant_sample
		cl::Buffer, // ant_allowed
		cl_int,     // problem_size
		cl_double,  // alpha
		cl::Buffer  // rng_seeds
	> advanceAntsCL;

	cl::Buffer pheromone_d;
	cl::Buffer visibility_d;
	cl::Buffer weights_d;
	cl::Buffer routes_d;
	cl::Buffer routes_length_d;
	cl::Buffer ant_sample_d;
	cl::Buffer ant_allowed_d;
	cl::Buffer rng_seeds_d;

	Graph<int> allowed_data;

	void advanceAnts() {
		cl::NDRange global_size(problem.size());
		advanceAntsCL(
			cl::EnqueueArgs(queue, global_size),
			pheromone_d,
			visibility_d,
			weights_d,
			routes_d,
			routes_length_d,
			ant_sample_d,
			ant_allowed_d,
			problem.size(),
			params.alpha,
			rng_seeds_d			
		).wait();
	}

public:
	static constexpr const char* static_name = "manyant";
	static constexpr const char* static_params = "";

	ManyAntOptimizer(Problem problem, AntParams params)
	:	CLColonyOptimizer::CLColonyOptimizer(problem, params),
		advanceAntsCL(cl::Kernel()),
		pheromone(problem.size(), params.initial_pheromone) {}

	Graph<double> pheromone;

	void prepare() override {
		setupCL(false);
		program = loadProgramVariant(static_name);

		pheromone_d = createAndFillBuffer(problem.sizeSqr(), false, pheromone);
		weights_d = createAndFillBuffer(problem.sizeSqr(), false, problem.weights);
		routes_d = createAndFillBuffer<int>(problem.sizeSqr(), false, 0);
		routes_length_d = createAndFillBuffer(problem.size(), false, std::numeric_limits<cl_int>::max());
		ant_sample_d = createAndFillBuffer<double>(problem.sizeSqr(), false, 0.0);

		Graph<double> visibility = getVisibility();
		visibility_d = createAndFillBuffer(problem.sizeSqr(), false, visibility);

		allowed_data = getAllowedData();
		ant_allowed_d = createAndFillBuffer(problem.sizeSqr(), false, allowed_data);

		std::vector<uint> rngs = getRngs();
		rng_seeds_d = createAndFillBuffer(problem.size(), false, rngs);

		queue.finish();

		advanceAntsCL = decltype(advanceAntsCL)(cl::Kernel(program, "wander_ant"));
	}

	void optimize(unsigned int rounds) override {
		std::vector<int> ant_route_lengths(problem.size());
		std::vector<int> ant_route(problem.size());
		while (rounds-- > 0) {
			Profiler::start("opts");

			// TODO : Not 1:1 same rng as sequential because sequential reseeds rng every round
			
			Profiler::start("adva");
			advanceAnts();
			Profiler::stop("adva");

			Profiler::start("eval");
			queue.enqueueReadBuffer(routes_length_d, CL_TRUE, 0, sizeof(int) * ant_route_lengths.size(), ant_route_lengths.data());
			//l::copy(queue, routes_length_d, ant_route_lengths.begin(), ant_route_lengths.end());
			auto best_ant_it = std::min_element(ant_route_lengths.begin(), ant_route_lengths.end());
			size_t best_ant_idx = std::distance(ant_route_lengths.begin(), best_ant_it);
			queue.enqueueReadBuffer(routes_d, CL_TRUE, best_ant_idx * problem.size() * sizeof(int), sizeof(int) * problem.size(), ant_route.data());
			best_route_length = std::min(*best_ant_it, best_route_length);
			Profiler::stop("eval");
			

			Profiler::start("upda");
			for (auto& value : pheromone.adjacency_matrix.data) {
				value *= (1.0 - params.rho);
			}

			// Lay pheromone along the route of the best ant
			
			if (*best_ant_it < std::numeric_limits<int>::max()) {
				double spread = params.q / *best_ant_it;
				for (auto it = std::next(ant_route.begin()); it != ant_route.end(); it++) {
					auto prev = std::prev(it);
					pheromone.edge(*prev, *it) += spread;
				}
			}
		
			// clamp all pheromone values
			for (auto& value : pheromone.adjacency_matrix.data) {
				value = std::clamp(value, params.min_pheromone, params.max_pheromone);
			}

			queue.enqueueWriteBuffer(
				pheromone_d, CL_TRUE, 0,
				sizeof(double) * pheromone.adjacency_matrix.data.size(),
				pheromone.adjacency_matrix.data.data());
			queue.enqueueWriteBuffer(
				ant_allowed_d, CL_TRUE, 0,
				sizeof(int) * allowed_data.adjacency_matrix.data.size(),
				allowed_data.adjacency_matrix.data.data());
			Profiler::stop("upda");

			Profiler::stop("opts");
		}
	}
};

