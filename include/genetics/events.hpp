// =============================================================================
// events.hpp — Fine-grained event / hook system for the simulation loop.
//
// Allows users to inject callbacks at every phase of the generation cycle
// without modifying the Simulation class.  Useful for:
//   - Logging / progress bars
//   - Collecting time-series data
//   - Implementing custom logic (e.g., environmental change, migration pulses,
//     artificial selection, phenotype recording)
//   - Debugging / assertions
//
// Usage:
//   SimulationEvents<Pop> events;
//   events.on_pre_selection = [](Generation g, const Pop& p) { ... };
//   events.on_post_mutation = [](Generation g, const Pop& p) { ... };
//   events.on_generation_end = [](Generation g, const Pop& p) { ... };
//
//   AdvancedSimulation<Pop> sim(pop, seed, events);
//   sim.run(100);
//
// Events are optional — null callbacks are skipped with zero overhead.
// =============================================================================
#pragma once

#include "types.hpp"

#include <functional>

namespace gensim {

// ── SimulationEvents ────────────────────────────────────────────────────────
// A bundle of optional callbacks, one per simulation phase.
//
// All callbacks receive:
//   - Generation g   — current generation index (0-based)
//   - const Pop& pop — read-only reference to the population
//
// For callbacks that need to *modify* the population (e.g., migration pulses,
// environmental changes), use on_pre_selection_mut or on_post_advance_mut
// which receive a mutable reference.
// ─────────────────────────────────────────────────────────────────────────────
template <typename PopType>
struct SimulationEvents {
    // ── Const callbacks (observation / logging) ─────────────────────────────

    /// Fired at the very start of a generation, before fitness evaluation.
    std::function<void(Generation, const PopType&)> on_generation_start;

    /// Fired after fitness has been evaluated for all individuals.
    std::function<void(Generation, const PopType&)> on_post_fitness;

    /// Fired after parent selection and recombination, before mutation.
    std::function<void(Generation, const PopType&)> on_post_recombination;

    /// Fired after mutations have been applied to the offspring buffer.
    std::function<void(Generation, const PopType&)> on_post_mutation;

    /// Fired after buffers have been swapped (offspring → parents).
    /// This is equivalent to the existing GenerationCallback but placed in
    /// a structured events object.
    std::function<void(Generation, const PopType&)> on_generation_end;

    // ── Mutable callbacks (intervention) ────────────────────────────────────
    // Use these to implement per-generation modifications such as:
    //   - Injecting migrants
    //   - Changing selection coefficients (fluctuating environments)
    //   - Imposing artificial selection / culling
    //   - Triggering population resize for demographic events

    /// Mutable access before selection begins (e.g., change fitness model).
    std::function<void(Generation, PopType&)> on_pre_selection_mut;

    /// Mutable access after the new generation is in place.
    std::function<void(Generation, PopType&)> on_post_advance_mut;

    // ── Predicate: should the simulation stop early? ────────────────────────
    /// Return true to abort the run.  Checked at the end of each generation.
    /// Useful for convergence criteria, fixation detection, etc.
    std::function<bool(Generation, const PopType&)> should_stop;

    // ── Convenience ─────────────────────────────────────────────────────────
    /// True if no callbacks are registered (fast path).
    [[nodiscard]] bool empty() const noexcept {
        return !on_generation_start
            && !on_post_fitness
            && !on_post_recombination
            && !on_post_mutation
            && !on_generation_end
            && !on_pre_selection_mut
            && !on_post_advance_mut
            && !should_stop;
    }
};

// ── DataRecorder ────────────────────────────────────────────────────────────
// A pre-built event listener that accumulates per-generation measurements
// into vectors for later analysis or plotting.
//
// Usage:
//   DataRecorder<Pop> rec;
//   rec.track_mean_fitness = true;
//   rec.track_segregating_sites = true;
//   rec.sample_interval = 5;  // record every 5th generation
//   events.on_generation_end = rec.as_callback();
// ─────────────────────────────────────────────────────────────────────────────
template <typename PopType>
struct DataRecorder {
    // ── What to track ───────────────────────────────────────────────────────
    bool track_mean_fitness       = true;
    bool track_segregating_sites  = true;

    /// Record every `sample_interval` generations (1 = every generation).
    Generation sample_interval = 1;

    // ── Accumulated data ────────────────────────────────────────────────────
    std::vector<Generation> generations;
    std::vector<double>     mean_fitness;
    std::vector<std::size_t> segregating_sites;

    /// Build a callback suitable for SimulationEvents::on_generation_end.
    [[nodiscard]] std::function<void(Generation, const PopType&)>
    as_callback()
    {
        return [this](Generation g, const PopType& pop) {
            if (sample_interval > 0 && ((g + 1) % sample_interval != 0))
                return;
            generations.push_back(g);
            if (track_mean_fitness)
                mean_fitness.push_back(pop.mean_fitness());
            if (track_segregating_sites)
                segregating_sites.push_back(pop.segregating_sites());
        };
    }

    /// Reset all accumulated data.
    void clear() {
        generations.clear();
        mean_fitness.clear();
        segregating_sites.clear();
    }
};

}  // namespace gensim
