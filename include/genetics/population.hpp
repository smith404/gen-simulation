// =============================================================================
// population.hpp — Diploid population container, templated on storage backend
//                  and policy objects.
//
// Template parameters:
//   Storage          – DenseHaplotypes or SparseVariants (or any type that
//                      satisfies the HaplotypeStorage concept).
//   MutationPolicy   – callable (storage, table, gen, rng)
//   RecombPolicy     – callable (parents, offspring, h0, h1, ohap, rng)
//   FitnessPolicy    – callable (storage, individual, table) → Fitness
//
// The Population owns:
//   • the haplotype storage (with double-buffering),
//   • the global AlleleTable (shared across the simulation),
//   • a fitness cache (recomputed each generation before selection).
//
// Individual metadata is intentionally thin — we store only an index into the
// storage.  Users who need richer metadata (sex, age, spatial location) can
// extend Individual or keep a parallel vector.
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "types.hpp"

#include <cassert>
#include <numeric>
#include <random>
#include <vector>

namespace gensim {

// ── Thin per-individual metadata ────────────────────────────────────────────
struct Individual {
    std::size_t index = 0;   // index into the storage (haplotypes 2*i, 2*i+1)
    // Extend here: sex, age, location, pedigree pointers, etc.
};

// ── Population ──────────────────────────────────────────────────────────────
template <typename Storage,
          typename MutationPolicy,
          typename RecombPolicy,
          typename FitnessPolicy>
class Population {
public:
    // ── Construction ────────────────────────────────────────────────────────
    Population(std::size_t N, std::size_t L,
               MutationPolicy  mut_policy,
               RecombPolicy    rec_policy,
               FitnessPolicy   fit_policy,
               AlleleTable&    allele_table)
        : storage_{N, L}
        , mut_policy_{std::move(mut_policy)}
        , rec_policy_{std::move(rec_policy)}
        , fit_policy_{std::move(fit_policy)}
        , allele_table_{allele_table}
    {
        individuals_.resize(N);
        for (std::size_t i = 0; i < N; ++i) individuals_[i].index = i;
        fitness_cache_.resize(N, 1.0);
    }

    // ── Accessors ───────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t          size()     const noexcept { return individuals_.size(); }
    [[nodiscard]] const Storage&       storage()  const noexcept { return storage_; }
    [[nodiscard]]       Storage&       storage()        noexcept { return storage_; }
    [[nodiscard]] const AlleleTable&   alleles()  const noexcept { return allele_table_; }
    [[nodiscard]]       AlleleTable&   alleles()        noexcept { return allele_table_; }
    [[nodiscard]] const std::vector<Fitness>& fitness() const noexcept { return fitness_cache_; }

    // ── Policy access (for inspection / hot-swap) ───────────────────────────
    [[nodiscard]] const MutationPolicy& mutation_policy() const noexcept { return mut_policy_; }
    [[nodiscard]] const RecombPolicy&   recomb_policy()   const noexcept { return rec_policy_; }
    [[nodiscard]] const FitnessPolicy&  fitness_policy()  const noexcept { return fit_policy_; }

    // ── Core lifecycle (called by Simulation) ───────────────────────────────

    /// Recompute fitness for every individual.  This fills fitness_cache_.
    void compute_fitness() {
        const std::size_t N = individuals_.size();
        for (std::size_t i = 0; i < N; ++i) {
            fitness_cache_[i] = fit_policy_(storage_, i, allele_table_);
        }
    }

    /// Select a parent index by fitness-proportionate sampling (roulette wheel).
    /// Expects compute_fitness() to have been called this generation.
    [[nodiscard]] std::size_t select_parent(std::mt19937_64& rng) const {
        // Build a CDF on the fly.  For large N, caching the CDF or using
        // alias sampling would be more efficient.
        // PERF: precompute cumulative sum once per generation.
        std::discrete_distribution<std::size_t> dist(
            fitness_cache_.begin(), fitness_cache_.end());
        return dist(rng);
    }

    /// Produce one offspring haplotype by recombining two parental haplotypes
    /// and write it into the offspring buffer at position `offspring_hap`.
    void recombine(std::size_t parent_ind,
                   std::size_t offspring_hap,
                   std::mt19937_64& rng) {
        // The two parental haplotypes.
        std::size_t h0 = 2 * parent_ind;
        std::size_t h1 = 2 * parent_ind + 1;
        rec_policy_(storage_, storage_, h0, h1, offspring_hap, rng);
    }

    /// Apply mutations to all offspring haplotypes in the back buffer.
    void mutate(Generation gen, std::mt19937_64& rng) {
        mut_policy_(storage_, allele_table_, gen, rng);
    }

    /// Swap the offspring buffer to become the current generation.
    void advance() {
        storage_.swap_buffers();
    }

    /// Prepare offspring buffer for sparse backend.
    void prepare_offspring_buffer() {
        if constexpr (requires { storage_.clear_offspring_buffer(); }) {
            storage_.clear_offspring_buffer();
        }
    }

    // ── Statistics helpers ──────────────────────────────────────────────────

    [[nodiscard]] double mean_fitness() const noexcept {
        if (fitness_cache_.empty()) return 0.0;
        double sum = std::accumulate(fitness_cache_.begin(),
                                     fitness_cache_.end(), 0.0);
        return sum / static_cast<double>(fitness_cache_.size());
    }

    [[nodiscard]] std::size_t segregating_sites() const {
        return storage_.count_segregating_sites();
    }

private:
    Storage                 storage_;
    MutationPolicy          mut_policy_;
    RecombPolicy            rec_policy_;
    FitnessPolicy           fit_policy_;
    AlleleTable&            allele_table_;
    std::vector<Individual> individuals_;
    std::vector<Fitness>    fitness_cache_;
};

}  // namespace gensim
