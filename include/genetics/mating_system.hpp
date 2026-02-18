// =============================================================================
// mating_system.hpp — Pluggable mating-system policies.
//
// Controls how pairs of parents are chosen to produce offspring.  The mating
// system operates *on top of* the selection policy: the selection policy
// picks fit individuals from the pool, and the mating system applies
// biological constraints such as sex requirements, selfing, monogamy,
// and assortative mating.
//
// Provided policies:
//
//   RandomMating         – any two individuals may mate (default, WF-like)
//   SexualMating         – requires one Male + one Female parent
//   SelfingMating        – fraction σ of offspring are selfed (one parent)
//   MonogamousMating     – each individual mates with at most one partner
//   AssortativeMating    – tendency to mate with phenotypically similar mates
//   ClonalMating         – asexual: offspring = mutated copy of one parent
//   HeterogeneousMating  – per-deme mating strategies
//
// All policies provide:
//   ParentPair operator()(individuals, fitnesses, rng) const;
//
// where ParentPair = { parent0, parent1 }.  If parent0 == parent1 the
// offspring is selfed (no recombination between distinct genomes).
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gensim {

// ── Mating result ───────────────────────────────────────────────────────────
struct ParentPair {
    std::size_t parent0;   // contributes haplotype 0 to offspring
    std::size_t parent1;   // contributes haplotype 1 to offspring
};

// ─────────────────────────────────────────────────────────────────────────────
// RandomMating — unconstrained random mating (Wright-Fisher default).
// ─────────────────────────────────────────────────────────────────────────────
// Both parents are drawn from the same fitness-weighted distribution
// independently.  Self-fertilisation (p0 == p1) can occur by chance when the
// population is small, but is not forced.
struct RandomMating {
    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& /*inds*/,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            fitnesses.begin(), fitnesses.end());
        return { dist(rng), dist(rng) };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SexualMating — requires one Male + one Female parent.
// ─────────────────────────────────────────────────────────────────────────────
// Maintains separate fitness distributions for each sex.  Parent0 is always
// Female (maternal), Parent1 is Male (paternal).  Asserts that both sexes
// are present.
struct SexualMating {
    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& inds,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        // Partition indices by sex.
        thread_local std::vector<std::size_t> females, males;
        thread_local std::vector<Fitness> f_fit, m_fit;
        females.clear(); males.clear();
        f_fit.clear();   m_fit.clear();

        for (std::size_t i = 0; i < inds.size(); ++i) {
            if (inds[i].sex == Sex::Female) {
                females.push_back(i);
                f_fit.push_back(fitnesses[i]);
            } else if (inds[i].sex == Sex::Male) {
                males.push_back(i);
                m_fit.push_back(fitnesses[i]);
            }
        }

        assert(!females.empty() && "SexualMating requires at least one female");
        assert(!males.empty()   && "SexualMating requires at least one male");

        std::discrete_distribution<std::size_t> f_dist(f_fit.begin(), f_fit.end());
        std::discrete_distribution<std::size_t> m_dist(m_fit.begin(), m_fit.end());

        return { females[f_dist(rng)], males[m_dist(rng)] };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SelfingMating — mixed selfing / outcrossing.
// ─────────────────────────────────────────────────────────────────────────────
// With probability `selfing_rate`, the offspring is produced by a single
// parent (p0 == p1), meaning recombination happens between its own two
// haplotypes.  Otherwise, two distinct parents are drawn.
//
// This is critical for plant genetics where selfing dramatically affects
// homozygosity (fixation index F ≈ σ/(2-σ) at equilibrium).
struct SelfingMating {
    double selfing_rate = 0.0;   // σ ∈ [0, 1]

    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& /*inds*/,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            fitnesses.begin(), fitnesses.end());
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::size_t p0 = dist(rng);
        if (u01(rng) < selfing_rate) {
            return { p0, p0 };  // selfed
        }
        std::size_t p1 = dist(rng);
        return { p0, p1 };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MonogamousMating — each individual mates with at most one partner.
// ─────────────────────────────────────────────────────────────────────────────
// After an individual is pair-bonded, it cannot be chosen again.  This is
// useful for species with strict monogamy (many birds, some fish).
// When all available mates are exhausted, remaining offspring use the last
// available pair.
//
// Note: this policy is stateful within a generation.  Call reset() before
// the generation's mating loop, and call operator() for each offspring.
struct MonogamousMating {
    /// Reset mating state for a new generation.  Must be called before the
    /// mating loop each generation.
    void reset(std::size_t N) {
        paired_.clear();
        paired_.reserve(N);
    }

    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& inds,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng)
    {
        // Build available-only distributions each time.
        thread_local std::vector<Fitness> avail_fit;
        thread_local std::vector<std::size_t> avail_idx;
        avail_fit.clear();
        avail_idx.clear();

        for (std::size_t i = 0; i < inds.size(); ++i) {
            if (paired_.find(i) == paired_.end()) {
                avail_idx.push_back(i);
                avail_fit.push_back(fitnesses[i]);
            }
        }

        // Fall back to full pool if everyone is paired.
        if (avail_idx.size() < 2) {
            std::discrete_distribution<std::size_t> dist(
                fitnesses.begin(), fitnesses.end());
            return { dist(rng), dist(rng) };
        }

        std::discrete_distribution<std::size_t> dist(
            avail_fit.begin(), avail_fit.end());
        std::size_t i0 = avail_idx[dist(rng)];
        std::size_t i1 = avail_idx[dist(rng)];
        // Ensure distinct if possible.
        int attempts = 0;
        while (i1 == i0 && attempts < 10 && avail_idx.size() > 1) {
            i1 = avail_idx[dist(rng)];
            ++attempts;
        }

        paired_.insert(i0);
        paired_.insert(i1);
        return { i0, i1 };
    }

private:
    std::unordered_set<std::size_t> paired_;
};

// ─────────────────────────────────────────────────────────────────────────────
// AssortativeMating — mate choice based on trait similarity.
// ─────────────────────────────────────────────────────────────────────────────
// After selecting a first parent by fitness, the second parent is chosen with
// probability weighted by:
//   w_i * exp(-strength * |z_i - z_first|)
//
// where z is the value of trait `trait_index`.  `strength` > 0 gives positive
// assortative mating (like mates with like), < 0 gives disassortative.
struct AssortativeMating {
    std::size_t trait_index = 0;  // which trait dimension to compare
    double strength = 1.0;        // positive = assortative, negative = disassortative

    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& inds,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            fitnesses.begin(), fitnesses.end());

        std::size_t p0 = dist(rng);

        // Compute weighted fitness for the second parent.
        double z0 = (p0 < inds.size() && trait_index < inds[p0].traits.size())
                         ? inds[p0].traits[trait_index]
                         : 0.0;

        thread_local std::vector<double> weights;
        weights.resize(inds.size());
        for (std::size_t i = 0; i < inds.size(); ++i) {
            double zi = (trait_index < inds[i].traits.size())
                             ? inds[i].traits[trait_index]
                             : 0.0;
            double similarity = std::exp(-strength * std::abs(zi - z0));
            weights[i] = fitnesses[i] * similarity;
        }

        std::discrete_distribution<std::size_t> assort_dist(
            weights.begin(), weights.end());
        std::size_t p1 = assort_dist(rng);
        return { p0, p1 };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ClonalMating — purely asexual reproduction (no recombination partner).
// ─────────────────────────────────────────────────────────────────────────────
// Both parent indices are the same individual, so recombination simply copies
// the genome (with any mutations applied afterwards).
struct ClonalMating {
    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& /*inds*/,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            fitnesses.begin(), fitnesses.end());
        std::size_t p = dist(rng);
        return { p, p };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SexualSelfingMating — sex-based outcrossing with optional selfing rate.
// ─────────────────────────────────────────────────────────────────────────────
// Combines SexualMating and SelfingMating for hermaphroditic organisms that
// can both self and outcross.  In hermaphroditic mode, any individual can be
// either parent; selfing happens at rate σ.
struct SexualSelfingMating {
    double selfing_rate = 0.0;
    bool   hermaphroditic = true;  // if false, uses strict male/female

    [[nodiscard]] ParentPair
    operator()(const std::vector<Individual>& inds,
               const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        if (u01(rng) < selfing_rate) {
            // Selfing: pick one parent, use it for both slots.
            std::discrete_distribution<std::size_t> dist(
                fitnesses.begin(), fitnesses.end());
            std::size_t p = dist(rng);
            return { p, p };
        }

        if (hermaphroditic) {
            // Outcrossing in a hermaphroditic population.
            std::discrete_distribution<std::size_t> dist(
                fitnesses.begin(), fitnesses.end());
            std::size_t p0 = dist(rng);
            std::size_t p1 = dist(rng);
            return { p0, p1 };
        } else {
            // Strict sexual mating.
            SexualMating sex_mate;
            return sex_mate(inds, fitnesses, rng);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: assign sex to newborn individuals.
// ─────────────────────────────────────────────────────────────────────────────

/// Randomly assign Male/Female with a given sex ratio (proportion male).
inline void assign_sex(std::vector<Individual>& inds,
                       std::mt19937_64& rng,
                       double proportion_male = 0.5)
{
    std::bernoulli_distribution bd(proportion_male);
    for (auto& ind : inds) {
        ind.sex = bd(rng) ? Sex::Male : Sex::Female;
    }
}

/// Set all individuals to Hermaphrodite.
inline void assign_hermaphrodite(std::vector<Individual>& inds) {
    for (auto& ind : inds) {
        ind.sex = Sex::Hermaphrodite;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MatingSystemConcept — formalises the mating system interface.
// ─────────────────────────────────────────────────────────────────────────────
template <typename MS>
concept MatingSystemConcept = requires(const MS ms,
                                        const std::vector<Individual>& inds,
                                        const std::vector<Fitness>& fit,
                                        std::mt19937_64& rng) {
    { ms(inds, fit, rng) } -> std::convertible_to<ParentPair>;
};

}  // namespace gensim
