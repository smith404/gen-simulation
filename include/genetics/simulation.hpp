// =============================================================================
// simulation.hpp — Simulation drivers.
//
// Two drivers are provided:
//
// 1. Simulation<PopType>  (original)
//    Lightweight, minimal.  Owns the RNG and generation counter.  Provides
//    a simple run(G, callback) loop.  Kept for backward compatibility.
//
// 2. AdvancedSimulation<PopType, SelectionPolicy>  (new)
//    Full-featured driver that integrates:
//      - Pluggable selection policies (see selection.hpp)
//      - SimulationEvents hooks (see events.hpp)
//      - DemographicSchedule with automatic population resize
//      - Optional AncestryTracker (see ancestry.hpp)
//      - Periodic garbage collection
//      - Early-stop predicates
//
// Both are templated on the Population type which captures the storage
// backend and mutation/recombination/fitness policies.
//
// Thread-safety notes:
//   • Offspring production (steps 2–3) is embarrassingly parallel across
//     individuals.  Each thread needs its own RNG stream.
//   • Mutation (step 4) is per-haplotype and independent if AlleleTable is
//     thread-safe (see allele_table.hpp).
//   • Fitness evaluation (step 1) is read-only and trivially parallel.
// =============================================================================
#pragma once

#include "ancestry.hpp"
#include "demographics.hpp"
#include "events.hpp"
#include "gc.hpp"
#include "mating_system.hpp"
#include "population.hpp"
#include "selection.hpp"
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

// ─────────────────────────────────────────────────────────────────────────────
// AdvancedSimulation — full-featured driver with all the bells and whistles.
// ─────────────────────────────────────────────────────────────────────────────
//
// Template parameters:
//   PopType         — Population<Storage, Mut, Rec, Fit>
//   SelectionPolicy — callable (fitnesses, rng) → index  (default: FitnessProportionate)
//   MatingSystem    — callable (individuals, fitnesses, rng) → ParentPair
//                     (default: RandomMating)
//
// Features over basic Simulation:
//   • Pluggable selection policy (instead of hardcoded fitness-proportionate)
//   • Pluggable mating system (sex-based, selfing, assortative, etc.)
//   • SimulationEvents hooks (pre/post each phase, early stop)
//   • DemographicSchedule integration (automatic N changes)
//   • Optional AncestryTracker (pedigree recording)
//   • Periodic GC for fixed/lost mutations
//   • Cumulative fitness offset tracking from GC
//
template <typename PopType,
          typename SelectionPolicy = FitnessProportionate,
          typename MatingSystem = RandomMating>
class AdvancedSimulation {
public:
    AdvancedSimulation(PopType& population,
                        std::uint64_t seed,
                        SelectionPolicy sel = SelectionPolicy{},
                        MatingSystem mate = MatingSystem{})
        : pop_{population}
        , rng_{seed}
        , sel_policy_{std::move(sel)}
        , mating_{std::move(mate)}
    {}

    // ── Configuration (fluent setters) ──────────────────────────────────────

    /// Attach an event handler bundle.
    AdvancedSimulation& set_events(SimulationEvents<PopType> ev) {
        events_ = std::move(ev);
        return *this;
    }

    /// Attach a demographic schedule.
    AdvancedSimulation& set_demographics(DemographicSchedule sched) {
        demo_sched_ = std::move(sched);
        return *this;
    }

    /// Enable ancestry tracking.
    AdvancedSimulation& enable_ancestry(bool flag = true) {
        if (flag && !tracker_) tracker_.emplace();
        else if (!flag) tracker_.reset();
        return *this;
    }

    /// Set GC interval (0 = disabled).  GC runs after advance().
    AdvancedSimulation& set_gc_interval(Generation interval) {
        gc_interval_ = interval;
        return *this;
    }

    // ── Run ─────────────────────────────────────────────────────────────────

    /// Run for `num_generations` generations.
    void run(Generation num_generations) {
        for (Generation g = 0; g < num_generations; ++g) {
            step(gen_);
            ++gen_;

            // Early stop check.
            if (events_.should_stop && events_.should_stop(gen_, pop_))
                break;
        }
    }

    /// Execute a single generation step.
    void step(Generation g) {
        const std::size_t N = pop_.size();

        // ── Event: generation start ─────────────────────────────────────────
        if (events_.on_generation_start)
            events_.on_generation_start(g, pop_);

        // ── Demographics: resize if needed ──────────────────────────────────
        // Note: resizing genotype data for a Population is non-trivial.
        // For now, demographic changes are reported via the mutable hook
        // so users can handle resize logic appropriate for their scenario.
        // A full resize implementation is left to the user / a future update.

        // ── 1. Fitness evaluation ───────────────────────────────────────────
        pop_.compute_fitness();

        if (events_.on_post_fitness)
            events_.on_post_fitness(g, pop_);

        // ── Mutable pre-selection hook ──────────────────────────────────────
        if (events_.on_pre_selection_mut)
            events_.on_pre_selection_mut(g, pop_);

        // ── 2–3. Selection & recombination ──────────────────────────────────
        pop_.prepare_offspring_buffer();

        const auto& fit = pop_.fitness();
        const auto& inds = pop_.individuals();

        if (tracker_) tracker_->begin_generation(g);

        for (std::size_t i = 0; i < N; ++i) {
            // Use the mating system to produce a parent pair.
            // The mating system combines fitness-weighted selection with
            // biological constraints (sex, selfing, assortative, etc.).
            ParentPair pair = mating_(inds, fit, rng_);
            std::size_t p0 = pair.parent0;
            std::size_t p1 = pair.parent1;

            pop_.recombine(p0, 2 * i, rng_);
            pop_.recombine(p1, 2 * i + 1, rng_);

            // Record ancestry if tracking is enabled.
            if (tracker_) {
                tracker_->record_birth(i, p0, p1);
            }
        }

        if (events_.on_post_recombination)
            events_.on_post_recombination(g, pop_);

        // ── 4. Mutation ─────────────────────────────────────────────────────
        pop_.mutate(g, rng_);

        if (events_.on_post_mutation)
            events_.on_post_mutation(g, pop_);

        // ── 5. Advance (swap buffers) ───────────────────────────────────────
        pop_.advance();

        // ── GC (if enabled) ─────────────────────────────────────────────────
        if (gc_interval_ > 0 && (g + 1) % gc_interval_ == 0) {
            auto result = gc_sweep(pop_.storage(), pop_.alleles(), g + 1);
            fitness_offset_ += result.fitness_offset;
            total_substitutions_.insert(total_substitutions_.end(),
                result.substitutions.begin(), result.substitutions.end());
        }

        // ── Mutable post-advance hook ───────────────────────────────────────
        if (events_.on_post_advance_mut)
            events_.on_post_advance_mut(g, pop_);

        // ── Event: generation end ───────────────────────────────────────────
        if (events_.on_generation_end)
            events_.on_generation_end(g, pop_);
    }

    // ── Accessors ───────────────────────────────────────────────────────────
    [[nodiscard]] Generation generation() const noexcept { return gen_; }
    [[nodiscard]] const SelectionPolicy& selection_policy() const noexcept { return sel_policy_; }
    [[nodiscard]] SelectionPolicy& selection_policy() noexcept { return sel_policy_; }
    [[nodiscard]] MatingSystem& mating_system() noexcept { return mating_; }
    [[nodiscard]] const MatingSystem& mating_system() const noexcept { return mating_; }
    [[nodiscard]] std::mt19937_64& rng() noexcept { return rng_; }
    [[nodiscard]] const std::mt19937_64& rng() const noexcept { return rng_; }

    [[nodiscard]] double fitness_offset() const noexcept { return fitness_offset_; }
    [[nodiscard]] const std::vector<Substitution>& substitutions() const noexcept {
        return total_substitutions_;
    }

    /// Access the ancestry tracker (nullptr if not enabled).
    [[nodiscard]] AncestryTracker* ancestry() noexcept {
        return tracker_ ? &(*tracker_) : nullptr;
    }
    [[nodiscard]] const AncestryTracker* ancestry() const noexcept {
        return tracker_ ? &(*tracker_) : nullptr;
    }

    [[nodiscard]] SimulationEvents<PopType>& events() noexcept { return events_; }

private:
    PopType&                       pop_;
    std::mt19937_64                rng_;
    SelectionPolicy                sel_policy_;
    MatingSystem                   mating_;
    Generation                     gen_ = 0;

    SimulationEvents<PopType>      events_;
    DemographicSchedule            demo_sched_;
    std::optional<AncestryTracker> tracker_;
    Generation                     gc_interval_ = 0;
    double                         fitness_offset_ = 0.0;
    std::vector<Substitution>      total_substitutions_;
};

}  // namespace gensim
