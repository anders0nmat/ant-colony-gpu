#pragma once

#include <CL/opencl.hpp>
#include <iostream>

#include "../optimizer.hpp"

class CLColonyOptimizer: public AntOptimizer {
protected:
	std::string loadFile(std::filesystem::path path) {
		std::ifstream file(path);
		return std::string(
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

	cl::Program loadProgram(std::filesystem::path path) {
		std::string program_source = loadFile(path);
		cl::Program program(context, program_source);
		if (program.build() != CL_SUCCESS) {
			std::cerr
				<< "[OpenCL] Error building program \"" << path.filename() << "\":"
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			exit(EXIT_FAILURE);
		}
		return program;
	}

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
public:
	static constexpr const char* static_name = "opencl";
	static constexpr const char* static_params = "";

	using AntOptimizer::AntOptimizer;
};