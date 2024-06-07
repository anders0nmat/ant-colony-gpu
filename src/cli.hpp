#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

struct CliParameters {
private:
	enum class ArgumentType { parameter, flag };
	struct Argument {
		std::string name;
		std::unordered_set<std::string> aliases;
		std::string description;

		ArgumentType type;
		std::string value;
		bool isSet;

		Argument(std::string name, std::unordered_set<std::string> aliases, std::string description, std::string value) 
		: name(name), aliases(aliases), description(description), type(ArgumentType::parameter), value(value), isSet(false) {}
		
		Argument(std::string name, std::unordered_set<std::string> aliases, std::string description, bool isSet) 
		: name(name), aliases(aliases), description(description), type(ArgumentType::flag), value(""), isSet(isSet) {}

		~Argument() {}
	};

	std::vector<Argument> arguments;
	std::unordered_map<std::string, size_t> aliases;
	std::vector<std::string> items;
public:
	void addParameter(std::string name, std::string description,  std::unordered_set<std::string> alias = {}, std::string _default = "") {
		arguments.emplace_back(Argument(name, alias, description, _default));
		aliases[name] = arguments.size() - 1;
		addAlias(name, alias);
	}

	void addFlag(std::string name, std::string description, std::unordered_set<std::string> alias = {}) {
		arguments.emplace_back(Argument(name, alias, description, false));	
		aliases[name] = arguments.size() - 1;
		addAlias(name, alias);
	}

	void addAlias(std::string original, std::string alias) {
		aliases[alias] = aliases.at(original);
	}

	void addAlias(std::string original, std::unordered_set<std::string> alias) {
		size_t originalIndex = aliases.at(original);
		for (auto a : alias) {
			aliases[a] = originalIndex;
		}
	}

	std::string param(std::string name) {
		size_t index = aliases.at(name);
		Argument& arg = arguments.at(index);
		if (arg.type != ArgumentType::parameter) {
			throw std::invalid_argument("Argument was not registered as parameter");
		}
		return arg.value;
	}

	bool flag(std::string name) {
		size_t index = aliases.at(name);
		Argument& arg = arguments.at(index);
		if (arg.type != ArgumentType::flag) {
			throw std::invalid_argument("Argument was not registered as flag");
		}
		return arg.isSet;
	}

	void parse(int argc, char* argv[]) {
		for (int i = 1; i < argc; i++) {
			std::string arg (argv[i]);
			if (arg.front() == '-') {
				// argument
				auto namePos = arg.find_first_not_of('-');
				if (namePos > 2) {
					throw std::invalid_argument("Argument starts with more than two dashes: " + arg);
				}
				auto eqPos = arg.find_first_of('=');

				auto argName = arg.substr(namePos, eqPos == std::string::npos ? eqPos : eqPos - namePos);
				auto index = aliases.find(argName);
				if (index == aliases.end()) {
					throw std::invalid_argument("Unknown Argument: " + arg);
				}
				Argument& argument = arguments.at(index->second);
				switch (argument.type) {
					case ArgumentType::flag:
						argument.isSet = true;
						break;
					case ArgumentType::parameter:
						size_t eqPos = arg.find_first_of('=');
						if (eqPos != std::string::npos) {
							std::string value = arg.substr(eqPos + 1);
							argument.value = value;
						}
						else {
							i++;
							if (i >= argc) {
								throw std::invalid_argument("No value provided for parameter: " + arg);
							}
							arg = argv[i];
							argument.value = arg;
						}
						break;
				}
			}
			else {
				// entry
				items.push_back(arg);
			}
		}
	}

	std::vector<std::string>& entries() {
		return items;
	}

	std::string help(int num_columns = 2, int column_width = 8) {
		std::string result;
		for (auto& argument : arguments) {
			std::vector<std::string> al (argument.aliases.begin(), argument.aliases.end());
			std::sort(al.begin(), al.end());
			int column = 0;
			for (auto& alias : al) {
				if (alias.size() > column_width) {
					result += alias + "\n";
					column = 0;
				}
				else {
					result += std::string(column_width - alias.size(), ' ') + alias + "   ";
					column++;
				}

				if (column >= num_columns) {
					result += "\n";
					column = 0;
				}
			}

			int remaining_space = (num_columns - column) * column_width + (num_columns - column - 1) * 3;

			if (argument.name.size() > remaining_space) {
				result += (column == 0 ? "" : "\n") + argument.name + "\n" + std::string(remaining_space, ' ');
			}
			else {
				result += std::string(remaining_space - argument.name.size(), ' ') + argument.name;
			}

			result += " : " + argument.description;
			result += "\n";
		}
		return result;
	}
};