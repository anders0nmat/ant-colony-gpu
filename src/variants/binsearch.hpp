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

	std::vector<int> allowed_data;

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
		pheromone(problem.size(), params.initial_pheromone),
		visibility(problem.size()) {}

	Graph<double> pheromone;
	Graph<double> visibility;

	void prepare() override {
		setupCL(true);
		program = loadProgram("./src/variants/binsearch.cl");

		pheromone_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * pheromone.adjacency_matrix.data.size());
		visibility_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * visibility.adjacency_matrix.data.size());
		weights_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_length_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.size());
		ant_sample_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * problem.weights.adjacency_matrix.data.size());
		ant_allowed_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		ant_allowed_template_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		rng_seeds_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * problem.size());
		probabilities_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * problem.sizeSqr());

		std::transform(problem.weights.adjacency_matrix.data.cbegin(), problem.weights.adjacency_matrix.data.cend(),
			visibility.adjacency_matrix.data.begin(), [this](const int& w) {
				double visibility = 1.0 / std::max(params.zero_weight, static_cast<double>(w));
				return std::pow(visibility, params.beta);
			} );

		std::vector<int> allowed_prototype(problem.size(), 0);
		for (int i = 0; i < problem.dependencies.adjacency_matrix.dimension; i++) {
			int acc = 0;
			for (int j = 0; j < problem.dependencies.adjacency_matrix.dimension; j++) {
				if (problem.dependencies.edge(i, j)) {
					acc++;
				}
			}
			allowed_prototype.at(i) = acc;
		}
		for (size_t from = 0; from < problem.size(); from++) {
			if (problem.dependencies.edge(from, 0)) {
				allowed_prototype.at(from) -= 1;
			}
		}
		allowed_prototype.at(0) = -1;
		allowed_data.resize(problem.size() * problem.size(), 0);
		for (int i = 0; i < allowed_data.size(); i++) {
			allowed_data.at(i) = allowed_prototype.at(i % allowed_prototype.size());
		}

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
		std::fill(zero_ints.begin(), zero_ints.end(), std::numeric_limits<cl_int>::max());
		queue.enqueueWriteBuffer(
			routes_length_d, CL_FALSE, 0,
			sizeof(int) * problem.size(),
			zero_ints.data());
		std::fill(zero_ints.begin(), zero_ints.end(), 0);
		queue.enqueueWriteBuffer(
			ant_sample_d, CL_FALSE, 0,
			sizeof(double) * zero_doubles.size(),
			zero_doubles.data());
		/*queue.enqueueWriteBuffer(
			ant_allowed_d, CL_FALSE, 0,
			sizeof(int) * allowed_data.size(),
			allowed_data.data());*/
		queue.enqueueWriteBuffer(
			ant_allowed_template_d, CL_FALSE, 0,
			sizeof(int) * allowed_data.size(),
			allowed_data.data());
		std::vector<uint> rngs(problem.size());
		std::minstd_rand0 rng(params.random_seed);
		for (auto& i : rngs) {
			i = rng();
		}
		queue.enqueueWriteBuffer(
			rng_seeds_d, CL_FALSE, 0,
			sizeof(uint) * rngs.size(),
			rngs.data());
		queue.enqueueWriteBuffer(
			probabilities_d, CL_FALSE, 0,
			sizeof(double) * zero_doubles.size(),
			zero_doubles.data());
		queue.finish();

		advanceAntsCL = cl::KernelFunctor<
			cl::Buffer, // probabilities
			cl::Buffer, // weights
			cl::Buffer, // ant_routes
			cl::Buffer, // ant_routes_length
			cl::Buffer, // ant_sample
			cl::Buffer, // ant_allowed
			cl_int,     // problem_size
			cl::Buffer  // rng_seeds
		>(cl::Kernel(program, "wander_ant"));

		updatePheromoneCL = cl::KernelFunctor<
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
		>(cl::Kernel(program, "update_pheromone"));
	
		resetAllowedCL = cl::KernelFunctor<
			cl::Buffer, // allowed_template
			cl::Buffer  // allowed_data
		>(cl::Kernel(program, "reset_allowed"));


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

