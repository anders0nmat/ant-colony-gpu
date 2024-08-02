#pragma once

#include <algorithm>
#include <random>

#include "clcolony.hpp"
#include "../profiler.hpp"

class SampleMaskOptimizer: public CLColonyOptimizer {
protected:
	cl::Program program;
	cl::KernelFunctor<
		cl::Buffer, // probabilities
		cl::Buffer, // weights
		cl::Buffer, // dependencies
		cl::Buffer, // ant_need_visit
		cl::Buffer, // ant_routes
		cl::Buffer, // ant_routes_length
		cl::Buffer, // ant_sample
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
		cl::Buffer // ant_need_visit
	> resetAntNeedVisitCL;

	cl::Buffer pheromone_d;
	cl::Buffer visibility_d;
	cl::Buffer weights_d;
	cl::Buffer routes_d;
	cl::Buffer routes_length_d;
	cl::Buffer ant_sample_d;
	cl::Buffer rng_seeds_d;
	cl::Buffer probabilities_d;
	cl::Buffer dependencies_d;
	cl::Buffer ant_need_visit_d;

	void advanceAnts() {
		cl::NDRange global_size(problem.size());
		advanceAntsCL(
			cl::EnqueueArgs(queue, global_size),
			probabilities_d,
			weights_d,
			dependencies_d,
			ant_need_visit_d,
			routes_d,
			routes_length_d,
			ant_sample_d,
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

	void resetAntNeedVisit() {
		cl::NDRange global_size(bitmask_size);
		resetAntNeedVisitCL(
			cl::EnqueueArgs(queue, global_size),
			ant_need_visit_d
		).wait();
	}

public:
	static constexpr const char* static_name = "samplemask";
	static constexpr const char* static_params = "";

	SampleMaskOptimizer(Problem problem, AntParams params)
	:	CLColonyOptimizer::CLColonyOptimizer(problem, params),
		advanceAntsCL(cl::Kernel()),
		updatePheromoneCL(cl::Kernel()),
		resetAntNeedVisitCL(cl::Kernel()),
		pheromone(problem.size(), params.initial_pheromone) {}

	Graph<double> pheromone;
	bool forceInt32Bitmasks = false;
	int bitmask_size = 0;

	void prepare() override {
		setupCL(false);
		program = loadProgramVariant(static_name);
		
		pheromone_d = createAndFillBuffer(problem.sizeSqr(), false, pheromone);
		weights_d = createAndFillBuffer(problem.sizeSqr(), true, problem.weights);
		routes_d = createAndFillBuffer<int>(problem.sizeSqr(), false, 0);
		routes_length_d = createAndFillBuffer(problem.size(), false, std::numeric_limits<cl_int>::max());
		ant_sample_d = createAndFillBuffer<double>(problem.sizeSqr(), false, 0.0);
		probabilities_d = createAndFillBuffer<double>(problem.sizeSqr(), false, 0.0);


		const int mask_bit_size = 32;
		const int req_bitmask_fields = problem.size() / mask_bit_size + (problem.size() % mask_bit_size != 0 ? 1 : 0);
		std::vector<cl_uint> dep_mask = getDependencyMask(true);

		// How to properly check whether device supports int64?
		// FULL_PROFILE must support int64 i think...
		if (!forceInt32Bitmasks && device.getInfo<CL_DEVICE_PROFILE>() == "FULL_PROFILE") {
			std::vector<cl_ulong> dep_mask_long = getLongDependencyMask(dep_mask);
			dependencies_d = createAndFillBuffer(dep_mask_long.size(), true, dep_mask_long);

			bitmask_size = dep_mask_long.size();

			int req_ulong_bitmask_fields = req_bitmask_fields / 2 + (req_bitmask_fields % 2 != 0 ? 1 : 0);
			ant_need_visit_d = createBuffer<cl_ulong>(req_ulong_bitmask_fields * problem.size(), false);
		}
		else {
			dependencies_d = createAndFillBuffer(dep_mask.size(), true, dep_mask);
			
			bitmask_size = dep_mask.size();

			ant_need_visit_d = createBuffer<cl_uint>(req_bitmask_fields * problem.size(), false);
		}

		Graph<double> visibility = getVisibility();
		visibility_d = createAndFillBuffer(problem.sizeSqr(), true, visibility);

		std::vector<uint> rngs = getRngs();
		rng_seeds_d = createAndFillBuffer(problem.size(), false, rngs);
		queue.finish();

		advanceAntsCL = decltype(advanceAntsCL)(cl::Kernel(program, "wander_ant"));
		updatePheromoneCL = decltype(updatePheromoneCL)(cl::Kernel(program, "update_pheromone"));	
		resetAntNeedVisitCL = decltype(resetAntNeedVisitCL)(cl::Kernel(program, "reset_ant_need_visit"));

		resetAntNeedVisit();
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
			resetAntNeedVisit();
			Profiler::stop("upda");

			Profiler::stop("opts");
		}
	}
};

