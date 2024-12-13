#pragma once

#include <CL/opencl.hpp>
#include <iostream>
#include <cassert>

#include "../optimizer.hpp"

class CLColonyOptimizer: public AntOptimizer {
protected:
	std::string loadFileString(std::filesystem::path path) {
		std::ifstream file(path);
		return std::string(
			std::istreambuf_iterator<char>(file),
			std::istreambuf_iterator<char>());
	}

	std::vector<char> loadFileBinary(std::filesystem::path path) {
		std::ifstream file(path, std::ios::binary);
		return std::vector<char>(
			std::istreambuf_iterator<char>(file),
			std::istreambuf_iterator<char>());
	}

	void setupCL(bool verbose) {
		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		if (all_platforms.empty()) {
			std::cerr
				<< "[OpenCL] No platforms found" << std::endl;
			exit(EXIT_FAILURE);
		}
		cl::Platform default_platform = all_platforms.at(0);
		if (verbose) {
			std::cout
				<< "[OpenCL] Using platform: "
				<< default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
		}

		std::vector<cl::Device> devices;
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if (devices.empty()) {
			std::cerr << "[OpenCL] No devices found." << std::endl;
			exit(EXIT_FAILURE);
		}

		device = devices.at(0);
		if (verbose) {
			std::cout
				<< "[OpenCL] Using device: "
				<< device.getInfo<CL_DEVICE_NAME>() << "\n";
		}

		context = cl::Context(device);
		queue = cl::CommandQueue(context, device);
	}

	cl::Program loadBinaryProgram(std::filesystem::path path, std::string compiler_args) {
		std::vector<char> program_binary = loadFileBinary(path);
		cl::Program program(context, program_binary);

		cl_int succ = program.build(compiler_args);
		if (succ != CL_SUCCESS) {
			std::cerr
				<< "[OpenCL] Error creating program " << path.filename() << ": "
				<< "(" << succ << ")"
				<< "\n";
			exit(EXIT_FAILURE);
		}

		succ = program.build();
		if (succ != CL_SUCCESS) {
			std::cerr
				<< "[OpenCL] Error building program " << path.filename() << ": "
				<< "(" << succ << ") "
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			exit(EXIT_FAILURE);
		}
		return program;		
	}

	cl::Program loadTextProgram(std::filesystem::path path, std::string compiler_args) {
		std::string program_source = loadFileString(path);
		cl::Program program(context, program_source);
		std::filesystem::path location = path;
		location.remove_filename();

		cl_int succ = program.build("-I \"" + location.string() + "\" " + compiler_args);
		if (succ != CL_SUCCESS) {
			std::cerr
				<< "[OpenCL] Error building program " << path.filename() << ": "
				<< "(" << succ << ") "
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			exit(EXIT_FAILURE);
		}
		
		return program;
	}

	bool is_spirv_file(std::filesystem::path path) {
		std::ifstream file(path, std::ios::binary);
		union {
			char bytes[4];
			uint32_t number;
		} magic_number;
		file.read(&magic_number.bytes[0], 4);
		return magic_number.number == 0x07230203 || magic_number.number == 0x03022307;	
	}

	cl::Program loadProgram(std::filesystem::path path, std::string compiler_args = "") {
		if (is_spirv_file(path)) {
			return loadBinaryProgram(path, compiler_args);
		}
		else {
			return loadTextProgram(path, compiler_args);
		}
	}

	cl::Program loadProgramVariant(const char* variant_name, std::string compiler_args = "") {
		return loadProgram(
			std::filesystem::path("./src/variants") /
			std::filesystem::path(variant_name).replace_extension(".cl"),
			compiler_args
		);
	}

	template<typename T>
	cl::LocalSpaceArg createLocalBuffer(size_t size) {
		return cl::Local(sizeof(T) * size);
	}

	template<typename T>
	cl::Buffer createBuffer(size_t size, bool read_only) {
		return cl::Buffer(context, read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE, sizeof(T) * size);	
	}

	template<typename T>
	cl::Buffer createAndFillBuffer(size_t size, bool read_only, T content) {
		cl::Buffer result = createBuffer<T>(size, read_only);
		queue.enqueueFillBuffer(result, content, 0, sizeof(T) * size);
		return result;
	}

	template<typename T>
	cl::Buffer createAndFillBuffer(size_t size, bool read_only, const std::vector<T>& data) {
		assert(size == data.size());
		cl::Buffer result = createBuffer<T>(size, read_only);
		queue.enqueueWriteBuffer(
			result,
			CL_FALSE,
			0,
			sizeof(T) * size,
			data.data());
		return result;
	}
	
	template<typename T>
	cl::Buffer createAndFillBuffer(size_t size, bool read_only, const Graph<T>& data) {
		return createAndFillBuffer(size, read_only, data.adjacency_matrix.data);
	}

	// Commonly used prepare optimizations

	Graph<double> getVisibility() {
		Graph<double> result(problem.size());
		std::transform(problem.weights.adjacency_matrix.data.cbegin(), problem.weights.adjacency_matrix.data.cend(),
			result.adjacency_matrix.data.begin(), [this](const int& w) {
				double visibility = 1.0 / std::max(params.zero_weight, static_cast<double>(w));
				return std::pow(visibility, params.beta);
			} );
		return result;
	}

	std::vector<int> getAllowedList() {
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
		return allowed_prototype;	
	}

	Graph<int> getAllowedData() {
		std::vector<int> allowed_prototype = getAllowedList();
		Graph<int> result(problem.size(), 0);
		for (auto it = result.adjacency_matrix.data.begin(); it != result.adjacency_matrix.data.end(); std::advance(it, problem.size())) {
			std::copy(allowed_prototype.begin(), allowed_prototype.end(), it);
		}	
		return result;
	}

	std::vector<uint> getRngs() {
		std::vector<uint> rngs(problem.size(), 0);
		std::minstd_rand0 rng(params.random_seed);
		std::generate(rngs.begin(), rngs.end(), rng);
		return rngs;
	}

	/*
	@param swap : Swap from "dependent on idx" to "idx depends on"
	*/
	std::vector<cl_uint> getDependencyMask(bool swap) {
		const int bitmask_size = 32;
		const int req_bitmask_fields = problem.size() / bitmask_size + (problem.size() % bitmask_size != 0 ? 1 : 0);
		std::vector<cl_uint> dep_mask(problem.size() * req_bitmask_fields, 0);
		for (size_t i = 0; i < problem.size(); i++) {
			for (size_t j = 0; j < problem.size(); j++) {
				size_t idx = (swap ? i : j) * (req_bitmask_fields * bitmask_size) + (swap ? j : i);
				if (problem.dependencies.edge(i, j)) {
					dep_mask[idx / bitmask_size] |= (1 << idx % bitmask_size);
				}
			}
		}
		return dep_mask;
	}

	std::vector<cl_ulong> getLongDependencyMask(const std::vector<cl_uint> dep_mask) {
		const int bitmask_size = 32;
		const int req_bitmask_fields = problem.size() / bitmask_size + (problem.size() % bitmask_size != 0 ? 1 : 0);
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
		return dep_mask_long;
	}

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
public:
	static constexpr const char* static_name = "opencl";
	static constexpr const char* static_params = "";

	using AntOptimizer::AntOptimizer;
};