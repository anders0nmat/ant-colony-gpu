#pragma once

#include <random>

#include "../optimizer.hpp"

class SequentialOptimizer: public AntOptimizer {
public:
	struct Ant {
		int current_node = 0;
		std::vector<int> allowed_nodes;
		std::vector<int> route;
		int route_length = 0;

		struct minstd0_engine {
			u_int32_t state;

			uint32_t operator()() {
				const uint a = 16807;
				const uint c = 0;
				const uint m = 2147483647;

				state = (a * (state) + c) % m;

				return state;
			}

			void seed(uint32_t seed) {
				state = seed;
			}
		};

		minstd0_engine random_generator;
	};

	static constexpr const char* static_name = "sequential";
	static constexpr const char* static_params = "";

	using AntOptimizer::AntOptimizer;

	Graph<double> pheromone;
	Graph<double> visibility;

	Ant prototype_ant;
	std::minstd_rand0 random_generator;

	SequentialOptimizer(Problem problem, AntParams params)
	:	AntOptimizer::AntOptimizer(problem, params),
		pheromone(problem.size(), params.initial_pheromone),
		visibility(problem.size()) {}

	void prepare() override {
		prototype_ant.allowed_nodes.resize(problem.dependencies.adjacency_matrix.dimension, 0);
		prototype_ant.route.push_back(0);

		for (int i = 0; i < problem.dependencies.adjacency_matrix.dimension; i++) {
			int acc = 0;
			for (int j = 0; j < problem.dependencies.adjacency_matrix.dimension; j++) {
				if (problem.dependencies.edge(i, j)) {
					acc++;
				}
			}
			prototype_ant.allowed_nodes.at(i) = acc;
		}

		for (size_t from = 0; from < problem.size(); from++) {
			if (problem.dependencies.edge(from, 0)) {
				prototype_ant.allowed_nodes.at(from) -= 1;
			}
		}
		prototype_ant.allowed_nodes.at(0) = -1;

		// Precompute (Visibility)^(beta) because neither can change during the optimization
		auto it1 = problem.weights.adjacency_matrix.data.begin();
		auto it1end = problem.weights.adjacency_matrix.data.end();
		auto it2 = visibility.adjacency_matrix.data.begin();
		auto it2end = visibility.adjacency_matrix.data.end();
		for (; it1 != it1end && it2 != it2end; it1++, it2++) {
			double visibility = 1.0 / std::max(params.zero_weight, static_cast<double>(*it1));
			*it2 = std::pow(visibility, params.beta);
		}

		random_generator.seed(params.random_seed);
	}

	void optimize(unsigned int rounds) override {
		std::vector<Ant> ants(problem.size());
		for (Ant& ant : ants) {
			ant.random_generator.seed(random_generator());
		}
		while (rounds-- > 0) {
			Profiler::start("opts");

			Ant* best_ant = nullptr;

			Profiler::start("adva");
			// Let ants wander ; Keep track of the best ant & best route
			for (Ant& ant : ants) {
				// Init Ant
				ant.allowed_nodes = prototype_ant.allowed_nodes;
				ant.current_node = 0;
				ant.route = prototype_ant.route;
				ant.route_length = 0;
				//ant.random_generator.seed(random_generator());

				// Wander Ant
				for (int i = 0; i < problem.size() - 1; i++) {
					advance_ant(ant);
					if (ant.current_node < 0) { break; }
				} 
				
				// If ant not at end (== stuck)
				if (ant.current_node != problem.size() - 1) {
					continue;
				}

				// Calculate performance of ant (== route length)
				ant.route_length = problem.weights.route_length(ant.route.begin(), ant.route.end());
				best_route_length = std::min(best_route_length, ant.route_length);
				if (best_ant == nullptr || best_ant->route_length > ant.route_length) {
					best_ant = &ant;
				}
			}
			Profiler::stop("adva");

			Profiler::start("eval");
			// eval phase is integrated into advance phase... Just pretend it is instant i guess 
			Profiler::stop("eval");

			Profiler::start("upda");
			// Evaporate pheromone
			for (auto& value : pheromone.adjacency_matrix.data) {
				value *= (1.0 - params.rho);
			}

			// Lay pheromone along the route of the best ant
			if (best_ant != nullptr) {
				double spread = params.q / best_ant->route_length;
				for (auto it = std::next(best_ant->route.begin()); it != best_ant->route.end(); it++) {
					auto prev = std::prev(it);
					pheromone.edge(*prev, *it) += spread;
				}
			}

			// clamp all pheromone values
			for (auto& value : pheromone.adjacency_matrix.data) {
				value = std::clamp(value, params.min_pheromone, params.max_pheromone);
			}
			Profiler::stop("upda");

			Profiler::stop("opts");
		}
	}

private:
	double edge_value(size_t from, size_t to) {
		double pher = pheromone.edge(from, to);
		double vis = visibility.edge(from, to);
		return std::pow(pher, params.alpha) * vis;
	}

	void advance_ant(Ant& ant) {
		if (ant.current_node < 0) { return; }

		bool hasPossibleNext = false;
		std::vector<double> next_nodes(problem.size(), 0.0);
		double sum = 0.0;
		for (size_t next = 0; next < problem.size(); next++) {
			if (ant.allowed_nodes.at(next) != 0) { continue; }
			double val = edge_value(ant.current_node, next);
			next_nodes.at(next) = val;
			sum += val;
			hasPossibleNext = hasPossibleNext || next_nodes.at(next) > 0;
		}

		if (!hasPossibleNext) {
			ant.current_node = -1;
			return;
		}

		int next_node = -1;
		double rd = (static_cast<double>(ant.random_generator()) / UINT32_MAX) * sum;
		for (size_t i = 0; i < next_nodes.size(); i++) {
			rd -= next_nodes[i];
			if (rd < 0) {
				next_node = i;
				break;
			}
		}

		//std::discrete_distribution<size_t> dist(next_nodes.begin(), next_nodes.end());
		//size_t next_node = dist(ant.random_generator);
		if (next_node == -1) {
			ant.current_node = -1;
			return;
		}

		ant.current_node = next_node;
		ant.route.push_back(next_node);
		ant.allowed_nodes.at(next_node) = -1;
		for (size_t from = 0; from < problem.size(); from++) {
			if (problem.dependencies.edge(from, next_node)) {
				ant.allowed_nodes.at(from) -= 1;
			}
		}
	}

};