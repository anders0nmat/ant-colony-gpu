#pragma once

#include <CL/opencl.hpp>
#include <iostream>

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

	cl::Program loadBinaryProgram(std::filesystem::path path) {
		std::vector<char> program_binary = loadFileBinary(path);
		cl::Program program(context, program_binary);

		cl_int succ = program.build();
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

	cl::Program loadTextProgram(std::filesystem::path path) {
		std::string program_source = loadFileString(path);
		cl::Program program(context, program_source);
		std::filesystem::path location = path;
		location.remove_filename();

		cl_int succ = program.build("-I \"" + location.string() + "\"");
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

	cl::Program loadProgram(std::filesystem::path path) {
		if (is_spirv_file(path)) {
			return loadBinaryProgram(path);
		}
		else {
			return loadTextProgram(path);
		}
	}

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
public:
	static constexpr const char* static_name = "opencl";
	static constexpr const char* static_params = "";

	using AntOptimizer::AntOptimizer;
};