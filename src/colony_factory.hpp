#pragma once
#include <string>
#include <memory>
#include <unordered_map>

#include "problem.hpp"
#include "params.hpp"
#include "optimizer.hpp"

struct ColonyFactory {
	virtual std::pair<std::string, std::string> identifier() const = 0;
	std::string signature() {
		auto p = identifier();
		return p.second.empty() ? p.first : p.first + ":" + p.second;
	}
	virtual std::unique_ptr<AntOptimizer> make(const Problem& problem, AntParams params) = 0;
	virtual ~ColonyFactory() = default;
public:
	using ColonyList = std::unordered_map<std::string, std::unique_ptr<ColonyFactory>>;
	static ColonyList variants;

	template<typename Ty>
	static void add();

	static ColonyFactory* get(std::string identifier) {
		auto it = variants.find(identifier);
		if (it != variants.end()) {
			return it->second.get();
		}
		return nullptr;
	}
};

template<typename Ty>
struct ConcreteColonyFactory: ColonyFactory {
	std::pair<std::string, std::string> identifier() const override {
		return std::make_pair(Ty::static_name, Ty::static_params);
	}

	std::unique_ptr<AntOptimizer> make(const Problem& problem, AntParams params) override {
		std::unique_ptr<AntOptimizer> e = std::make_unique<Ty>(problem, params);
		return e;
	}
};

template<typename Ty>
void ColonyFactory::add() {
	std::unique_ptr<ColonyFactory> factory = std::make_unique<ConcreteColonyFactory<Ty>>();
	auto identifier = factory->identifier();
	variants[identifier.first] = std::move(factory);
}

