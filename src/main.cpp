#include <iostream>

#include "profiler.hpp"
#include "colony_factory.hpp"
#include "cli.hpp"

#include "variants/sequential.hpp"
#include "variants/manyant.hpp"
#include "variants/manyant2.hpp"
#include "variants/gpupher.hpp"
#include "variants/phercomp.hpp"
#include "variants/binsearch.hpp"
#include "variants/depmask.hpp"
#include "variants/samplemask.hpp"
#include "variants/parant.hpp"
#include "variants/parant2.hpp"
#include "variants/parant3.hpp"
#include "variants/parant4.hpp"
#include "variants/localant.hpp"
#include "variants/neighbor.hpp"
#include "variants/gpumax.hpp"


Profiler Profiler::default_profiler;
ColonyFactory::ColonyList ColonyFactory::variants;
CliParameters cli;

int main(int argc, char* argv[]) {
	ColonyFactory::add<SequentialOptimizer>();
	ColonyFactory::add<ManyAntOptimizer>();
	ColonyFactory::add<ManyAnt2Optimizer>();
	ColonyFactory::add<GpuPherOptimizer>();
	ColonyFactory::add<PherCompOptimizer>();
	ColonyFactory::add<BinSearchOptimizer>();
	ColonyFactory::add<DepMaskOptimizer>();
	ColonyFactory::add<SampleMaskOptimizer>();
	ColonyFactory::add<ParAntOptimizer>();
	ColonyFactory::add<ParAnt2Optimizer>();
	ColonyFactory::add<ParAnt3Optimizer>();
	ColonyFactory::add<ParAnt4Optimizer>();
	ColonyFactory::add<LocalAntOptimizer>();
	ColonyFactory::add<NeighborOptimizer>();
	ColonyFactory::add<GpuMaxOptimizer>();

	cli.addFlag("help", "Prints this help message", {"h"});
	cli.addFlag("list", "List all optimization variants available", {"l"});
	cli.addParameter("colony", "Selects the colony to run. Colony arguments are separated by a colon (:)", {"c"});
	cli.addParameter("rounds", "How many rounds of optimization should be run", {"r"}, "500");
	cli.addParameter("seed", "Controls the random-number-generator seed", {}, "thomas");

	cli.parse(argc, argv);

	if (cli.flag("help")) {
		std::cout 
			<< "Ant Colony Optimization -- OpenCL\n"
			<< "Usage:\n"
			<< "  main <problem.sop> [flags]\n\n"
			<< "Flags:"
			<< cli.help()
			<< std::endl;
		return EXIT_SUCCESS;
	}

	if (cli.flag("list")) {
		std::cout << "Available optimization variants:\n";
		for (const auto& e : ColonyFactory::variants) {
			std::cout << "  " << e.second->signature() << "\n";
		}
		std::cout << std::endl;
		return EXIT_SUCCESS;
	}

	if (cli.entries().size() != 1) {
		std::cerr
			<< (cli.entries().size() == 0 ? "Not enough" : "Too many")
			<< " files provided\n"
			<< "See --help for more information" << std::endl;
		return EXIT_FAILURE;
	}

	std::string colonyIdentifier = cli.param("colony");
	std::string colonyArguments;
	size_t argumentSep = colonyIdentifier.find_first_of(':');
	if (argumentSep != std::string::npos) {
		colonyArguments = colonyIdentifier.substr(argumentSep + 1);
		colonyIdentifier = colonyIdentifier.substr(0, argumentSep - 1);
	}
	
	ColonyFactory* factory = ColonyFactory::get(colonyIdentifier);
	if (factory == nullptr) {
		std::cerr
			<< "Unknown colony identifier: "
			<< "\"" << colonyIdentifier << "\""
			<< std::endl;
		return EXIT_FAILURE;
	}

	Problem problem(cli.entries().front());

	AntParams params;
	params.alpha = 0.5;
	params.beta = 0.5;
	params.q = 100;
	params.rho = 0.5;

	params.initial_pheromone = 1;
	params.min_pheromone = 0.01;
	params.max_pheromone = 100;

	params.zero_weight = 0.001;
	params.random_seed = std::hash<std::string>{}(cli.param("seed"));

	params.variant_args = colonyArguments;

	unsigned int rounds = std::stoul(cli.param("rounds"));

	std::unique_ptr<AntOptimizer> optimizer = factory->make(problem, params);

	Profiler::start("prep");
	optimizer->prepare();
	Profiler::stop("prep");

	Profiler::start("optr");
	optimizer->optimize(rounds);
	Profiler::stop("optr");

	auto basic_analysis = Profiler::analyze("opts");
	std::cout
		<< "Finished!\n"
		<< "Variant: " << colonyIdentifier << (colonyArguments.empty() ? "" : ":" + colonyArguments) << "\n"
		<< "Result length: " << optimizer->best_route_length << " (" << problem.solution_bounds.first << ", " << problem.solution_bounds.second << ")\n"
		<< "Prepare Time: " << Profiler::first("prep").value<double, std::milli>() << "ms\n"
		<< "Execution Time: " << Profiler::first("optr").value<double, std::milli>() << "ms\n"
		<< "Step Time:\n" 
			<< "  min: " << basic_analysis.min.value<double, std::milli>() << "ms\n"
			<< "  max: " << basic_analysis.max.value<double, std::milli>() << "ms\n"
			<< "  avg: " << basic_analysis.avg.value<double, std::milli>() << "ms\n";
		
	for (const auto& id : Profiler::measurement_keys()) {
		auto analysis = Profiler::analyze(id);
		std::cout
			<< "Measurement '" << id << "':\n"
			<< "  min: " << analysis.min.value<double, std::milli>() << "ms\n"
			<< "  max: " << analysis.max.value<double, std::milli>() << "ms\n"
			<< "  avg: " << analysis.avg.value<double, std::milli>() << "ms\n";
	}

	std::cout
		<< "Score: " << static_cast<double>(rounds) / Profiler::first("optr").value<double>()  << " RPS\n"
		<< std::endl;
	return EXIT_SUCCESS;
}