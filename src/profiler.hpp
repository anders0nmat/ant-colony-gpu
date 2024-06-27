#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>


struct Profiler {
	using Clock = std::chrono::high_resolution_clock;
	using Duration = Clock::duration;
	using Timepoint = Clock::time_point;
	using Identifier = std::string;

	struct Timer {
		Timepoint start_time;
		std::string comment = "";
		bool is_active = false;
	};

	struct Measurement {
		Duration duration;
		std::string comment;

		Measurement(Duration d, std::string c)
		: duration(d), comment(c) {}

		template<typename T, typename Resolution = std::ratio<1>>
		T value() {
			return std::chrono::duration_cast<std::chrono::duration<T, Resolution>>(duration).count();
		}
	};
	using MeasurementList = std::vector<Measurement>;

	std::unordered_map<Identifier, Timer> active_timers;
	std::unordered_map<Identifier, MeasurementList> measurements;

	void start_timer(const Identifier & id, std::string comment = "") {
		Timer& timer = active_timers[id];
		if (timer.is_active) { return; }

		timer.comment = comment;
		timer.is_active = true;
		timer.start_time = Clock::now();
	}

	void stop_timer(const Identifier & id, std::string comment = "") {
		Timepoint stop_time = Clock::now();
		Timer& timer = active_timers[id];
		if (!timer.is_active) { return; }

		measurements[id].emplace_back(
			stop_time - timer.start_time,
			comment.empty() ? timer.comment : comment
		);
		timer.is_active = false;
	}

	std::tuple<Duration, Duration, Duration> get_minmaxavg(const Identifier& id) {
		auto it = measurements.at(id).begin();
		auto itend = measurements.at(id).end();
		size_t count = measurements.at(id).size();
		Duration
			min = it->duration,
			max = it->duration,
			total = it->duration;
		it++;
		for (;it != itend; it++) {
			min = std::min(it->duration, min);
			max = std::max(it->duration, max);
			total += it->duration;
		}
		return std::make_tuple(min, max, total / count);
	}

	static Profiler default_profiler;

	static void start(const Identifier & id, std::string comment = "") {
		default_profiler.start_timer(id, comment);
	}

	static void stop(const Identifier & id, std::string comment = "") {
		default_profiler.stop_timer(id, comment);
	}

	static MeasurementList& at(const Identifier & id) {
		return default_profiler.measurements.at(id);
	}

	static Measurement& first(const Identifier & id) {
		return at(id).front();
	}


};
