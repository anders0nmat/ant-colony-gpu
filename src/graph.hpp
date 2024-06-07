#pragma once
#include "matrix.hpp"

template<typename T>
struct Graph {
public:
	Matrix<T> adjacency_matrix;

	Graph()
	: adjacency_matrix(0) {}

	Graph(size_t size, T initial_weight = T())
	: adjacency_matrix(size, initial_weight) {}

	typename Matrix<T>::reference edge(size_t from, size_t to) {
		return adjacency_matrix.at(from, to);
	}

	size_t size() const {
		return adjacency_matrix.dimension;
	}

	template<class ForwardIterator>
	T route_length(ForwardIterator start, ForwardIterator end) {
		if (start >= end) { return T(); }

		T acc = T();
		std::advance(start, 1);
		for (;start != end; start++) {
			ForwardIterator prev = std::prev(start);
			acc += edge(*prev, *start);
		}
		return acc;
	}
};
