// =============================================================================
// concepts.hpp — C++20 concepts that formalise every pluggable interface.
//
// These concepts serve three purposes:
//   1. Documentation — the required API surface is machine-readable.
//   2. Diagnostics  — template errors become short, actionable messages.
//   3. Extensibility — third-party types that satisfy a concept are
//                      automatically usable with the library.
//
// Concepts defined:
//   HaplotypeStorage       — any double-buffered genotype container
//   PointerAccessStorage   — storage with contiguous per-haplotype memory
//   ElementAccessStorage   — storage with per-element get/set
//   MutationPolicyFor<S>   — callable that mutates an offspring buffer
//   RecombinationPolicyFor<S> — callable that recombines parental haplotypes
//   FitnessPolicyFor<S>    — callable that evaluates individual fitness
//   SelectionPolicy        — callable that picks a parent index from fitnesses
//   DFEConcept             — callable that draws a selection coefficient
// =============================================================================
#pragma once

#include "types.hpp"

#include <concepts>
#include <cstddef>
#include <random>
#include <vector>

namespace gensim {

// Forward declaration (needed by FitnessPolicyFor).
class AlleleTable;

// ─────────────────────────────────────────────────────────────────────────────
// HaplotypeStorage — the minimal interface every storage backend must satisfy.
// ─────────────────────────────────────────────────────────────────────────────
template <typename S>
concept HaplotypeStorage = requires(S s, const S cs,
                                     std::size_t idx, SiteIndex site) {
    // Size queries
    { cs.num_individuals() } -> std::convertible_to<std::size_t>;
    { cs.num_sites()       } -> std::convertible_to<std::size_t>;
    { cs.num_haplotypes()  } -> std::convertible_to<std::size_t>;

    // Segment copy (parent front → offspring back)
    { s.copy_segment(idx, idx, site, site) };

    // Double-buffer swap
    { s.swap_buffers() };

    // Statistics hook
    { cs.count_segregating_sites() } -> std::convertible_to<std::size_t>;
};

// ─────────────────────────────────────────────────────────────────────────────
// PointerAccessStorage — contiguous per-haplotype memory (like DenseHaplotypes).
// Enables memcpy-based block copies and direct array iteration.
// ─────────────────────────────────────────────────────────────────────────────
template <typename S>
concept PointerAccessStorage = HaplotypeStorage<S> &&
    requires(S s, const S cs, std::size_t hap) {
        { cs.hap_ptr(hap)            } -> std::same_as<const AlleleID*>;
        { s.hap_ptr(hap)             } -> std::same_as<AlleleID*>;
        { s.offspring_hap_ptr(hap)   } -> std::same_as<AlleleID*>;
    };

// ─────────────────────────────────────────────────────────────────────────────
// ElementAccessStorage — per-element get/set (like SparseVariants, or Dense
// with the convenience wrappers added in this refactor).
// ─────────────────────────────────────────────────────────────────────────────
template <typename S>
concept ElementAccessStorage = HaplotypeStorage<S> &&
    requires(S s, const S cs, std::size_t hap, SiteIndex site, AlleleID a) {
        { cs.get(hap, site)                  } -> std::convertible_to<AlleleID>;
        { cs.offspring_get(hap, site)        } -> std::convertible_to<AlleleID>;
        { s.offspring_set(hap, site, a)      };
    };

// ─────────────────────────────────────────────────────────────────────────────
// MutationPolicyFor<S> — applies mutations to offspring haplotypes.
// ─────────────────────────────────────────────────────────────────────────────
template <typename M, typename S>
concept MutationPolicyFor = requires(const M m, S& storage,
                                      AlleleTable& table,
                                      Generation gen,
                                      std::mt19937_64& rng) {
    { m(storage, table, gen, rng) };
};

// ─────────────────────────────────────────────────────────────────────────────
// RecombinationPolicyFor<S> — recombines two parental haplotypes into one
// offspring haplotype.
// ─────────────────────────────────────────────────────────────────────────────
template <typename R, typename S>
concept RecombinationPolicyFor = requires(const R r,
                                           const S& parents, S& offspring,
                                           std::size_t hap,
                                           std::mt19937_64& rng) {
    { r(parents, offspring, hap, hap, hap, rng) };
};

// ─────────────────────────────────────────────────────────────────────────────
// FitnessPolicyFor<S> — evaluates the fitness of a single individual.
// ─────────────────────────────────────────────────────────────────────────────
template <typename F, typename S>
concept FitnessPolicyFor = requires(const F f, const S& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) {
    { f(storage, ind, table) } -> std::convertible_to<Fitness>;
};

// ─────────────────────────────────────────────────────────────────────────────
// SelectionPolicy — picks a parent index given a fitness vector.
// ─────────────────────────────────────────────────────────────────────────────
template <typename SP>
concept SelectionPolicyConcept = requires(const SP sp,
                                           const std::vector<Fitness>& fit,
                                           std::mt19937_64& rng) {
    { sp(fit, rng) } -> std::convertible_to<std::size_t>;
};

// ─────────────────────────────────────────────────────────────────────────────
// DFEConcept — draws a selection coefficient from a distribution.
// ─────────────────────────────────────────────────────────────────────────────
template <typename D>
concept DFEConcept = requires(D d, std::mt19937_64& rng) {
    { d(rng) } -> std::convertible_to<double>;
};

// Note: MatingSystemConcept is defined in mating_system.hpp (requires
// Individual, ParentPair), and QuantitativeTraitFor is defined in
// quantitative_trait.hpp (requires Individual, AlleleTable).

}  // namespace gensim
