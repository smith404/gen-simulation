// =============================================================================
// demographics.hpp — Demographic events and population-size schedules.
//
// Supports:
//   - Instantaneous size changes (bottlenecks, expansions)
//   - Exponential growth phases
//   - Arbitrary N(t) via a schedule of events
//
// Usage:
//   DemographicSchedule sched;
//   sched.add_size_change(50, 200);           // at gen 50, set N = 200
//   sched.add_exponential_growth(100, 0.01);  // from gen 100, grow at r=0.01
//   sched.add_size_change(200, 1000);         // at gen 200, snap back to 1000
//
// The Simulation driver calls sched.population_size(gen, current_N) each
// generation to determine N for the next generation.
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

namespace gensim {

// ── Event types ─────────────────────────────────────────────────────────────

/// Instantaneous population-size change.
struct SizeChangeEvent {
    Generation when;
    std::size_t new_N;
};

/// Begin exponential growth: N(t) = N_start * exp(rate * (t - when)).
/// Growth continues until the next event overrides it.
struct ExponentialGrowthEvent {
    Generation when;
    double     rate;       // per-generation growth rate (can be negative for decline)
    std::size_t base_N;    // N at the start of this growth phase (set automatically)
};

/// A bottleneck: reduce to bottleneck_N for `duration` generations, then
/// restore to the previous N.  Convenience wrapper for two SizeChangeEvents.
struct BottleneckEvent {
    Generation  when;
    std::size_t bottleneck_N;
    Generation  duration;
    std::size_t restore_N;     // N to restore after the bottleneck
};

using DemoEvent = std::variant<SizeChangeEvent, ExponentialGrowthEvent,
                                BottleneckEvent>;

// ── DemographicSchedule ─────────────────────────────────────────────────────

class DemographicSchedule {
public:
    DemographicSchedule() = default;

    /// Add an instantaneous size change at generation `when`.
    void add_size_change(Generation when, std::size_t new_N) {
        events_.push_back(SizeChangeEvent{when, new_N});
        sort_events();
    }

    /// Add exponential growth starting at generation `when`.
    /// `base_N` is the population size at the start of growth;
    /// if set to 0, it will use whatever N is current at that generation.
    void add_exponential_growth(Generation when, double rate,
                                 std::size_t base_N = 0)
    {
        events_.push_back(ExponentialGrowthEvent{when, rate, base_N});
        sort_events();
    }

    /// Add a bottleneck: shrink to bottleneck_N for `duration` generations
    /// starting at `when`, then restore to `restore_N`.
    void add_bottleneck(Generation when, std::size_t bottleneck_N,
                        Generation duration, std::size_t restore_N)
    {
        events_.push_back(BottleneckEvent{when, bottleneck_N, duration, restore_N});
        sort_events();
    }

    /// Query: what should the population size be at generation `gen`,
    /// given the current size `current_N`?
    ///
    /// Call this at the top of each generation step.  Returns current_N
    /// unchanged if no event applies.
    [[nodiscard]] std::size_t population_size(Generation gen,
                                               std::size_t current_N) const
    {
        std::size_t N = current_N;
        for (auto& ev : events_) {
            N = std::visit(Evaluator{gen, current_N}, ev);
            if (N != current_N) return std::max(N, std::size_t{1});
        }
        return std::max(N, std::size_t{1});
    }

    [[nodiscard]] bool empty() const noexcept { return events_.empty(); }

private:
    std::vector<DemoEvent> events_;

    void sort_events() {
        std::sort(events_.begin(), events_.end(), [](auto& a, auto& b) {
            auto when = [](auto& ev) -> Generation {
                return std::visit([](auto& e){ return e.when; }, ev);
            };
            return when(a) < when(b);
        });
    }

    struct Evaluator {
        Generation gen;
        std::size_t current_N;

        std::size_t operator()(const SizeChangeEvent& e) const {
            return (gen == e.when) ? e.new_N : current_N;
        }

        std::size_t operator()(const ExponentialGrowthEvent& e) const {
            if (gen < e.when) return current_N;
            std::size_t base = (e.base_N > 0) ? e.base_N : current_N;
            double dt = static_cast<double>(gen - e.when);
            double new_N = static_cast<double>(base) * std::exp(e.rate * dt);
            return static_cast<std::size_t>(std::max(new_N, 1.0));
        }

        std::size_t operator()(const BottleneckEvent& e) const {
            if (gen == e.when) return e.bottleneck_N;
            if (gen == e.when + e.duration) return e.restore_N;
            return current_N;
        }
    };
};

}  // namespace gensim
