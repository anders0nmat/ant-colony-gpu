#pragma once

#include <vector>
#include <stdexcept>

template<typename T>
struct Matrix {
public:
	using reference = typename std::vector<T>::reference;

	size_t dimension;
	std::vector<T> data;

	size_t linear_index(size_t x, size_t y) {
		if (x >= dimension) {
			throw std::out_of_range("x index out of bounds (" + std::to_string(x) + " >= " + std::to_string(dimension) + ")");
		}
		if (y >= dimension) {
			throw std::out_of_range("y index out of bounds (" + std::to_string(y) + " >= " + std::to_string(dimension) + ")");
		}

		return x * dimension + y;
	}

	Matrix(size_t dimension, T default_value = T())
	: dimension(dimension) {
		data.resize(dimension * dimension, default_value);
	}

	reference at(size_t x, size_t y) {
		return data.at(linear_index(x, y));
	}

	size_t size() const {
		return dimension;
	}
};
