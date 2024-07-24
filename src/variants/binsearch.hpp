#pragma once

#include <algorithm>
#include <random>

#include "clcolony.hpp"
#include "../profiler.hpp"

class BinSearchOptimizer: public CLColonyOptimizer {
protected:
	cl::Program program;
	cl::KernelFunctor<
		cl::Buffer, // probabilities
		cl::Buffer, // weights
		cl::Buffer, // ant_routes
		cl::Buffer, // ant_routes_length
		cl::Buffer, // ant_sample
		cl::Buffer, // ant_allowed
		cl_int,     // problem_size
		cl::Buffer  // rng_seeds
	> advanceAntsCL;

	cl::KernelFunctor<
		cl::Buffer, // pheromone
		cl::Buffer, // probabilities
		cl::Buffer, // visibility
		cl_double, // alpha
		cl_double, // one_minus_roh
		cl_double, // min_pheromone
		cl_double, // max_pheromone
		cl::Buffer, // ant_routes
		cl_uint, // best_ant_idx
		cl_double, // best_ant_pheromone
		cl_int  // problem_size
	> updatePheromoneCL;

	cl::KernelFunctor<
		cl::Buffer, // allowed_template
		cl::Buffer // allowed_data
	> resetAllowedCL;

	cl::Buffer pheromone_d;
	cl::Buffer visibility_d;
	cl::Buffer weights_d;
	cl::Buffer routes_d;
	cl::Buffer routes_length_d;
	cl::Buffer ant_sample_d;
	cl::Buffer ant_allowed_d;
	cl::Buffer ant_allowed_template_d;
	cl::Buffer rng_seeds_d;
	cl::Buffer probabilities_d;

	void advanceAnts() {
		cl::NDRange global_size(problem.size());
		advanceAntsCL(
			cl::EnqueueArgs(queue, global_size),
			probabilities_d,
			weights_d,
			routes_d,
			routes_length_d,
			ant_sample_d,
			ant_allowed_d,
			problem.size(),
			rng_seeds_d			
		).wait();
	}

	void updatePheromone(uint best_ant, double spread) {
		cl::NDRange global_size(problem.size() * problem.size());
		updatePheromoneCL(
			cl::EnqueueArgs(queue, global_size),
			pheromone_d,
			probabilities_d,
			visibility_d,
			params.alpha,
			1 - params.rho,
			params.min_pheromone,
			params.max_pheromone,
			routes_d,
			best_ant,
			spread,
			problem.size()
		).wait();
	}

	void resetAllowed() {
		cl::NDRange global_size(problem.size() * problem.size());
		resetAllowedCL(
			cl::EnqueueArgs(queue, global_size),
			ant_allowed_template_d,
			ant_allowed_d
		).wait();
	}

public:
	static constexpr const char* static_name = "binsearch";
	static constexpr const char* static_params = "";

	BinSearchOptimizer(Problem problem, AntParams params)
	:	CLColonyOptimizer::CLColonyOptimizer(problem, params),
		advanceAntsCL(cl::Kernel()),
		updatePheromoneCL(cl::Kernel()),
		resetAllowedCL(cl::Kernel()),
		pheromone(problem.size(), params.initial_pheromone) {}

	Graph<double> pheromone;

	void prepare() override {
		setupCL(true);
		program = loadProgramVariant(static_name);

		pheromone_d = createAndFillBuffer(problem.sizeSqr(), false, pheromone);
		weights_d = createAndFillBuffer(problem.sizeSqr(), true, problem.weights);
		routes_d = createAndFillBuffer<int>(problem.sizeSqr(), false, 0);
		routes_length_d = createAndFillBuffer(problem.size(), false, std::numeric_limits<cl_int>::max());
		ant_sample_d = createAndFillBuffer<double>(problem.sizeSqr(), false, 0.0);
		ant_allowed_d = createBuffer<int>(problem.sizeSqr(), false);
		probabilities_d = createAndFillBuffer<double>(problem.sizeSqr(), false, 0.0);

		Graph<double> visibility = getVisibility();
		visibility_d = createAndFillBuffer(problem.sizeSqr(), true, visibility);

		Graph<int> allowed_data = getAllowedData();
		ant_allowed_template_d = createAndFillBuffer(problem.sizeSqr(), true, allowed_data);

		std::vector<uint> rngs = getRngs();
		rng_seeds_d = createAndFillBuffer(problem.size(), false, rngs);
		queue.finish();

		advanceAntsCL = decltype(advanceAntsCL)(cl::Kernel(program, "wander_ant"));
		updatePheromoneCL = decltype(updatePheromoneCL)(cl::Kernel(program, "update_pheromone"));
		resetAllowedCL = decltype(resetAllowedCL)(cl::Kernel(program, "reset_allowed"));
		
		resetAllowed();
		updatePheromone(0, 0);
	}

	void optimize(unsigned int rounds) override {
		std::vector<int> ant_route_lengths(problem.size());
		std::vector<int> ant_route(problem.size());
		while (rounds-- > 0) {
			Profiler::start("opts");

			Profiler::start("adva");
			advanceAnts();
			Profiler::stop("adva");

			Profiler::start("eval");
			queue.enqueueReadBuffer(routes_length_d, CL_TRUE, 0, sizeof(int) * ant_route_lengths.size(), ant_route_lengths.data());
			auto best_ant_it = std::min_element(ant_route_lengths.begin(), ant_route_lengths.end());
			size_t best_ant_idx = std::distance(ant_route_lengths.begin(), best_ant_it);
			queue.enqueueReadBuffer(routes_d, CL_TRUE, best_ant_idx * problem.size() * sizeof(int), sizeof(int) * problem.size(), ant_route.data());
			best_route_length = std::min(*best_ant_it, best_route_length);
			Profiler::stop("eval");
			
			
			Profiler::start("upda");
			updatePheromone(best_ant_idx, params.q / *best_ant_it);
			resetAllowed();
			Profiler::stop("upda");

			Profiler::stop("opts");
		}
	}
};

