// =============================================================================
// simulation.hpp — Main simulation driver.
//
// Simulation is templated on the full Population type (which already captures
// the storage backend and all three policies).  It owns the RNG and the
// generation counter, and exposes a simple `run(G)` loop plus per-generation
// hooks for logging or custom actions.
//
// The generation loop:
//   1. compute_fitness()        — evaluate w_i for every individual
//   2. select parents           — fitness-proportionate sampling
//   3. produce offspring        — for each offspring individual, pick two
//                                 parents, recombine each parent's diploid
//                                 genome into one haplotype, producing two
//                                 haplotypes per offspring
//   4. mutate                   — apply mutations to the offspring buffer
//   5. swap_buffers / advance   — offspring become the new current generation
//
// Thread-safety notes:
//   • Offspring production (steps 2–3) is embarrassingly parallel across
//     individuals.  Each thread needs its own RNG stream; seed each with
//     `main_seed + thread_id` or use std::seed_seq.
//   • Mutation (step 4) writes to the offspring buffer; per-haplotype work is
//     independent if the AlleleTable is thread-safe (see allele_table.hpp).
//   • Fitness evaluation (step 1) is read-only and trivially parallel.
//
// A simple per-thread RNG example is included in the runnable demo.
// =============================================================================
#pragma once

#include "population.hpp"
#include "types.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace gensim {

// ── Callback type for per-generation reporting ──────────────────────────────
// Receives the generation number and a const reference to the population.
template <typename PopType>
using GenerationCallback = std::function<void(Generation, const PopType&)>;

// ── Simulation ──────────────────────────────────────────────────────────────
template <typename PopType>
class Simulation {
public:
    Simulation(PopType& population, std::uint64_t seed)
        : pop_{population}, rng_{seed}
    {}

    /// Run the simulation for `num_generations` generations.
    /// If a callback is provided, it is invoked at the end of each generation
    /// (after offspring have replaced parents).
    void run(Generation num_generations,
             GenerationCallback<PopType> callback = nullptr)
    {
        for (Generation g = 0; g < num_generations; ++g) {
            step(g);
            if (callback) callback(gen_, pop_);
            ++gen_;
        }
    }

    /// Execute a single generation step.
    void step(Generation g) {
        const std::size_t N = pop_.size();

        // ── 1. Fitness evaluation ───────────────────────────────────────────
        pop_.compute_fitness();

        // ── 2–3. Selection & recombination ──────────────────────────────────
        // Prepare offspring buffer (no-op for Dense, clears back for Sparse).
        pop_.prepare_offspring_buffer();

        // Precompute a discrete distribution for parent selection so we don't
        // rebuild it for every individual.  This is O(N) space and O(N) build.
        const auto& fit = pop_.fitness();
        std::discrete_distribution<std::size_t> parent_dist(
            fit.begin(), fit.end());

        for (std::size_t i = 0; i < N; ++i) {
            // Each offspring needs two haplotypes.  Each comes from one parent's
            // diploid genome, recombined.
            //
            // Parent for haplotype 0 of offspring i:
            std::size_t p0 = parent_dist(rng_);
            pop_.recombine(p0, 2 * i, rng_);

            // Parent for haplotype 1 of offspring i:
            std::size_t p1 = parent_dist(rng_);
            pop_.recombine(p1, 2 * i + 1, rng_);
        }

        // ── 4. Mutation ─────────────────────────────────────────────────────
        pop_.mutate(g, rng_);

        // ── 5. Advance (swap buffers) ───────────────────────────────────────
        pop_.advance();
    }

    [[nodiscard]] Generation generation() const noexcept { return gen_; }

    // ── Per-thread RNG example ──────────────────────────────────────────────
    // For parallel offspring production, create per-thread engines:
    //
    //   std::vector<std::mt19937_64> thread_rngs;
    //   for (int t = 0; t < num_threads; ++t)
    //       thread_rngs.emplace_back(base_seed + t);
    //
    // Then partition individuals across threads and use thread_rngs[t] in each.
    // Remember to merge AlleleTable mutations if each thread has a local table.

private:
    PopType&          pop_;
    std::mt19937_64   rng_;
    Generation        gen_ = 0;
};

}  // namespace gensim
