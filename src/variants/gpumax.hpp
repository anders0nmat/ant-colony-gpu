#pragma once

#include <algorithm>
#include <random>
#include <bitset>
#include <type_traits>

#include "clcolony.hpp"
#include "../profiler.hpp"

class GpuMaxOptimizer: public CLColonyOptimizer {
protected:
	cl::Program program;
	cl::KernelFunctor<
		cl::Buffer, // probabilities
		cl::Buffer, // weights
		cl::Buffer, // dependencies
		cl::Buffer, // ant_routes
		cl::Buffer, // ant_routes_length
		cl::LocalSpaceArg, // ant_sample
		cl::LocalSpaceArg, // ant_allowed
		cl::Buffer, // ant_allowed_template
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
		cl_int  // problem_size
	> updatePheromoneCL;

	cl::KernelFunctor<
		cl::Buffer, // ant_route_length
		cl_double, // pheromone
		cl_int // problem_size
	> getBestAntCL;

	cl::Buffer pheromone_d;
	cl::Buffer visibility_d;
	cl::Buffer weights_d;
	cl::Buffer routes_d;
	cl::Buffer routes_length_d;
	cl::LocalSpaceArg ant_sample_d;
	cl::LocalSpaceArg ant_allowed_d;
	cl::Buffer ant_allowed_template_d;
	cl::Buffer rng_seeds_d;
	cl::Buffer probabilities_d;
	cl::Buffer dependencies_d;

	cl::Buffer best_length_d;

	std::vector<int> allowed_data;

	size_t work_size = 0;

	size_t leftmost_one(size_t value) {
		int i = 0;
		for (; i < sizeof(size_t) * 8; i++) {
			if (value >> i == 0) { return i; }
		}
		return i;
	}

	void advanceAnts() {
		cl::NDRange global_size(work_size, problem.size());
		cl::NDRange local_size(work_size, 1);
		advanceAntsCL(
			cl::EnqueueArgs(queue, global_size, local_size),
			probabilities_d,
			weights_d,
			dependencies_d,
			routes_d,
			routes_length_d,
			ant_sample_d,
			ant_allowed_d,
			ant_allowed_template_d,
			problem.size(),
			rng_seeds_d			
		).wait();
	}

	void updatePheromone() {
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
			problem.size()
		).wait();
	}

	void getBestAnt() {
		cl::NDRange global_size(1);
		getBestAntCL(
			cl::EnqueueArgs(queue, global_size),
			routes_length_d,
			params.q,
			problem.size()	
		).wait();
	}

public:
	static constexpr const char* static_name = "gpumax";
	static constexpr const char* static_params = "";

	GpuMaxOptimizer(Problem problem, AntParams params)
	:	CLColonyOptimizer::CLColonyOptimizer(problem, params),
		advanceAntsCL(cl::Kernel()),
		updatePheromoneCL(cl::Kernel()),
		getBestAntCL(cl::Kernel()),
		pheromone(problem.size(), params.initial_pheromone),
		visibility(problem.size()) {}

	Graph<double> pheromone;
	Graph<double> visibility;
	bool forceInt32Bitmasks = false;

	void prepare() override {
		setupCL(true);
		program = loadProgram("./src/variants/gpumax.cl");

		work_size = 1UL << leftmost_one(problem.size() - 1);

		pheromone_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * pheromone.adjacency_matrix.data.size());
		visibility_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * visibility.adjacency_matrix.data.size());
		weights_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		routes_length_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * problem.size());
		ant_sample_d = cl::Local(sizeof(cl_double) * work_size);
		ant_allowed_d = cl::Local(sizeof(int) * problem.size());
		ant_allowed_template_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * problem.weights.adjacency_matrix.data.size());
		rng_seeds_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * problem.size());
		probabilities_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * problem.sizeSqr());

		const int bitmask_size = 32;
		const int req_bitmask_fields = problem.size() / bitmask_size + (problem.size() % bitmask_size != 0 ? 1 : 0);
		std::vector<cl_uint> dep_mask(problem.size() * req_bitmask_fields, 0);
		for (size_t i = 0; i < problem.size(); i++) {
			for (size_t j = 0; j < problem.size(); j++) {
				size_t idx = j * (req_bitmask_fields * bitmask_size) + i;
				if (problem.dependencies.edge(i, j)) {
					dep_mask[idx / bitmask_size] |= (1 << idx % bitmask_size);
				}
			}
		}

		// How to properly check whether device supports int64?
		// FULL_PROFILE must support int64 i think...
		if (!forceInt32Bitmasks && device.getInfo<CL_DEVICE_PROFILE>() == "FULL_PROFILE") {
			std::vector<cl_ulong> dep_mask_long;
			bool append_to_last = false;
			for (size_t i = 0; i < dep_mask.size(); i++) {
				if (!append_to_last) {
					dep_mask_long.push_back(dep_mask[i]);
					append_to_last = (i + 1) % req_bitmask_fields != 0;
				}
				else {
					dep_mask_long.back() |= static_cast<cl_ulong>(dep_mask[i]) << 32;
					append_to_last = false;
				}
			}

			dependencies_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ulong) * dep_mask_long.size());
			queue.enqueueWriteBuffer(
				dependencies_d, CL_FALSE, 0,
				sizeof(cl_ulong) * dep_mask_long.size(),
				dep_mask_long.data());
		}
		else {
			dependencies_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_uint) * dep_mask.size());
			queue.enqueueWriteBuffer(
				dependencies_d, CL_FALSE, 0,
				sizeof(cl_uint) * dep_mask.size(),
				dep_mask.data());
		}

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
		std::vector<double> zero_doubles2(problem.size() * work_size, 0.0);
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
			cl::Buffer, // dependencies
			cl::Buffer, // ant_routes
			cl::Buffer, // ant_routes_length
			cl::LocalSpaceArg, // ant_sample
			cl::LocalSpaceArg, // ant_allowed
			cl::Buffer,
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
			cl_int  // problem_size
		>(cl::Kernel(program, "update_pheromone"));

		getBestAntCL = cl::KernelFunctor<
			cl::Buffer, // ant_route_length,
			cl_double, // pheromone
			cl_int // problem_size
		>(cl::Kernel(program, "get_best_ant"));

		updatePheromone();
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
			getBestAnt();
			Profiler::stop("eval");
			
			
			Profiler::start("upda");
			updatePheromone();
			Profiler::stop("upda");

			Profiler::stop("opts");
		}
	}
};

