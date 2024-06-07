#pragma once

#include "params.hpp"
#include "problem.hpp"

class AntOptimizer {
protected:
	Problem problem;
	AntParams params;	
public:
	int best_route_length = std::numeric_limits<int>::max();

	AntOptimizer(const Problem& problem, AntParams params)
	: problem(problem), params(params) {}

	virtual ~AntOptimizer() = default;

	virtual void prepare() = 0;
	virtual void optimize(unsigned int rounds) = 0;

	static constexpr const char* static_name = "abstract";
	static constexpr const char* static_params = "";
};
