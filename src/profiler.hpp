#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>


struct Profiler {
	using Clock = std::chrono::high_resolution_clock;
	using Duration = Clock::duration;
	using Timepoint = Clock::time_point;

	struct Timer {
		Timepoint start_time;
		std::string comment = "";
		bool is_active = false;
	};

	std::unordered_map<std::string, Timer> active_timers;
	std::unordered_map<std::string, std::vector<std::pair<Duration, std::string>>> measurements;

	void start_timer(const std::string & id, std::string comment = "") {
		Timer& timer = active_timers[id];
		if (timer.is_active) { return; }

		timer.comment = comment;
		timer.is_active = true;
		timer.start_time = Clock::now();
	}

	void stop_timer(const std::string & id, std::string comment = "") {
		Timepoint stop_time = Clock::now();
		Timer& timer = active_timers[id];
		if (!timer.is_active) { return; }

		measurements[id].emplace_back(
			stop_time - timer.start_time,
			comment.empty() ? timer.comment : comment
		);
		timer.is_active = false;
	}

	std::tuple<Duration, Duration, Duration> get_minmaxavg(const std::string& id) {
		auto it = measurements.at(id).begin();
		auto itend = measurements.at(id).end();
		size_t count = measurements.at(id).size();
		Duration
			min = it->first,
			max = it->first,
			total = it->first;
		it++;
		for (;it != itend; it++) {
			min = std::min(it->first, min);
			max = std::max(it->first, max);
			total += it->first;
		}
		return std::make_tuple(min, max, total / count);
	}

	static Profiler default_profiler;

	static void start(const std::string & id, std::string comment = "") {
		default_profiler.start_timer(id, comment);
	}

	static void stop(const std::string & id, std::string comment = "") {
		default_profiler.stop_timer(id, comment);
	}
};
