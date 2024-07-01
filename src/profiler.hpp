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

		friend bool operator<(const Measurement& lhs, const Measurement& rhs) {
			return lhs.duration < rhs.duration;
		}
	};
	using MeasurementList = std::vector<Measurement>;

	struct Analysis {
		Measurement min;
		Measurement max;
		Measurement avg;

		Analysis(Measurement min, Measurement max, Measurement avg)
		: min(min), max(max), avg(avg) {}
	};

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

	Analysis get_analysis(const Identifier& id) {
		auto it = measurements.at(id).begin();
		auto itend = measurements.at(id).end();
		size_t count = measurements.at(id).size();
		Measurement
			min = *it,
			max = *it,
			average = Measurement(it->duration, "");
		it++;
		for (;it != itend; it++) {
			min = std::min(*it, min);
			max = std::max(*it, max);
			average.duration += it->duration;
		}
		average.duration /= count;
		return Analysis(min, max, average);
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

	static Analysis analyze(const Identifier & id) {
		return default_profiler.get_analysis(id);
	}

	static std::vector<Identifier> measurement_keys() {
		std::vector<Identifier> ids;
		for (const auto& p : default_profiler.measurements) {
			ids.push_back(p.first);
		}
		return ids;
	}
};
