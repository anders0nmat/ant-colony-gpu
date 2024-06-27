#pragma once

#include "graph.hpp"
#include <filesystem>
#include <fstream>

struct Problem {
private:
	bool try_read_key(std::string key, std::string input, std::string& output) {
		if (output.empty() && input.find(key) == 0) {
			auto pos = input.find_first_not_of(": \t", key.size());
			output = input.substr(pos);
			return true;
		}
		return false;
	}
public:
	std::string name;
	std::string comment;
	std::pair<int, int> solution_bounds;

	Graph<int> weights;
	Graph<bool> dependencies;

	Problem(std::filesystem::path path)
	: name(""), comment(""), solution_bounds(-1, -1) {		
		std::ifstream file(path);
		int count = -2;
		size_t line_no = 0;
		std::string s = "";
		for (std::string line; std::getline(file, line);) {
			if (try_read_key("NAME", line, name)) { continue; }
			if (try_read_key("COMMENT", line, comment)) { continue; }
			if (try_read_key("SOLUTION_BOUNDS", line, s)) {
				size_t comma_pos = s.find_first_of(",");
				if (comma_pos != std::string::npos) {
					int a = std::stoi(s);
					int b = std::stoi(s.substr(comma_pos + 1));
					solution_bounds = std::make_pair(a, b);
				}
				else {
					int a = std::stoi(s);
					solution_bounds = std::make_pair(a, a);
				}
				continue;
			}

			if (line == "EDGE_WEIGHT_SECTION") {
				count = -1;
				continue;
			}

			if (count == -1) {
				count = std::stoi(line);
				weights = Graph<int>(count);
				dependencies = Graph<bool>(count);
				continue;
			}

			if (count >= 0 && line_no < count) {
				for (size_t column_no = 0; column_no < count; column_no++) {
					size_t next_num;
					int n = std::stoi(line, &next_num);

					if (n == -1) {
						// Dependency
						dependencies.edge(line_no, column_no) = true;
						weights.edge(line_no, column_no) = -1;
					}
					else {
						weights.edge(line_no, column_no) = 
							(n == 1000000 ? std::numeric_limits<int>::max() : n);
					}

					line = line.substr(next_num);
				}

				line_no++;
				continue;
			}
		}
		
	}

	size_t size() const {
		return weights.size();	
	}

	size_t sizeSqr() const {
		return size() * size();
	}
};

