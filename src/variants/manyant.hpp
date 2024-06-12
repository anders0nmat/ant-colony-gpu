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
		pheromone(problem.size(), params.initial_pheromone),
		visibility(problem.size()) {}

	Graph<double> pheromone;
	Graph<double> visibility;

	void prepare() override {
		setupCL(true);
		program = loadProgram("./src/variants/manyant.clcpp");

		pheromone_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * pheromone.adjacency_matrix.data.size());
		visibility_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * visibility.adjacency_matrix.data.size());
		weights_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_length_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.size());
		ant_sample_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * problem.weights.adjacency_matrix.data.size());
		ant_allowed_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		rng_seeds_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * problem.size());

		std::transform(problem.weights.adjacency_matrix.data.cbegin(), problem.weights.adjacency_matrix.data.cend(),
			visibility.adjacency_matrix.data.begin(), [this](const int& w) {
				double visibility = 1.0 / std::max(params.zero_weight, static_cast<double>(w));
				return std::pow(visibility, params.beta);
			} );

		queue.enqueueWriteBuffer(
			pheromone_d, CL_FALSE, 0,
			sizeof(double) * pheromone.adjacency_matrix.data.size(),
			pheromone.adjacency_matrix.data.data());
		queue.enqueueWriteBuffer(
			visibility_d, CL_FALSE, 0,
			sizeof(double) * visibility.adjacency_matrix.data.size(),
			visibility.adjacency_matrix.data.data());
		queue.enqueueWriteBuffer(
			weights_d, CL_FALSE, 0,
			sizeof(int) * problem.weights.adjacency_matrix.data.size(),
			problem.weights.adjacency_matrix.data.data());
		std::vector<int> zero_ints(problem.weights.adjacency_matrix.data.size(), 0);
		std::vector<double> zero_doubles(problem.weights.adjacency_matrix.data.size(), 0.0);
		queue.enqueueWriteBuffer(
			routes_d, CL_FALSE, 0,
			sizeof(int) * zero_ints.size(),
			zero_ints.data());
		queue.enqueueWriteBuffer(
			routes_length_d, CL_FALSE, 0,
			sizeof(int) * problem.size(),
			zero_ints.data());
		queue.enqueueWriteBuffer(
			ant_sample_d, CL_FALSE, 0,
			sizeof(double) * zero_doubles.size(),
			zero_doubles.data());
		queue.enqueueWriteBuffer(
			ant_allowed_d, CL_FALSE, 0,
			sizeof(int) * zero_ints.size(),
			zero_ints.data());
		std::vector<uint> rngs(problem.size());
		std::minstd_rand0 rng(params.random_seed);
		for (auto& i : rngs) {
			i = rng();
		}
		queue.enqueueWriteBuffer(
			rng_seeds_d, CL_FALSE, 0,
			sizeof(uint) * rngs.size(),
			rngs.data());
		queue.finish();
	}

	void optimize(unsigned int rounds) override {
		std::vector<int> ant_route_lengths(problem.size());
		std::vector<int> ant_route(problem.size());
		while (rounds-- > 0) {
			Profiler::start("opts");

			// TODO : Not 1:1 same rng as sequential because sequential reseeds rng every round
			advanceAnts();

			queue.enqueueReadBuffer(routes_length_d, CL_TRUE, 0, sizeof(int) * ant_route_lengths.size(), ant_route_lengths.data());
			auto best_ant_it = std::min_element(ant_route_lengths.begin(), ant_route_lengths.end());
			size_t best_ant_idx = std::distance(best_ant_it, ant_route_lengths.begin());
			queue.enqueueReadBuffer(routes_d, CL_TRUE, best_ant_idx * problem.size() * sizeof(int), sizeof(int) * problem.size(), ant_route.data());
			best_route_length = std::min(*best_ant_it, best_route_length);

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

			Profiler::stop("opts");
		}
	}
};

