// =============================================================================
// selection.hpp — Pluggable parent-selection policies.
//
// Decouples the selection algorithm from Population and Simulation so that
// users can mix and match selection models without modifying the core loop.
//
// Every selection policy is a lightweight callable:
//   std::size_t operator()(const std::vector<Fitness>& fitnesses,
//                          std::mt19937_64& rng) const;
//
// Provided models:
//   FitnessProportionate  — roulette-wheel / Wright–Fisher (default)
//   TournamentSelection   — k-way tournament
//   TruncationSelection   — select uniformly from the top fraction
//   RankSelection         — linear rank-based probabilities
//   BoltzmannSelection    — softmax / temperature-scaled fitness
//
// All satisfy the SelectionPolicyConcept defined in concepts.hpp.
// =============================================================================
#pragma once

#include "concepts.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace gensim {

// ─────────────────────────────────────────────────────────────────────────────
// FitnessProportionate (roulette-wheel / Wright–Fisher)
// ─────────────────────────────────────────────────────────────────────────────
// The classic: probability of selection ∝ fitness.
// Precomputes a discrete distribution per call.  For repeated calls within a
// single generation, callers should cache the distribution (as Simulation does).
// ─────────────────────────────────────────────────────────────────────────────
struct FitnessProportionate {
    [[nodiscard]] std::size_t
    operator()(const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            fitnesses.begin(), fitnesses.end());
        return dist(rng);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TournamentSelection
// ─────────────────────────────────────────────────────────────────────────────
// Pick `k` individuals uniformly at random; the fittest wins.
// Larger k → stronger selection pressure.  k=1 is equivalent to random mating.
// ─────────────────────────────────────────────────────────────────────────────
struct TournamentSelection {
    std::size_t tournament_size = 3;

    [[nodiscard]] std::size_t
    operator()(const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        assert(!fitnesses.empty());
        std::uniform_int_distribution<std::size_t> idx(0, fitnesses.size() - 1);
        std::size_t best = idx(rng);
        for (std::size_t t = 1; t < tournament_size; ++t) {
            std::size_t challenger = idx(rng);
            if (fitnesses[challenger] > fitnesses[best])
                best = challenger;
        }
        return best;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TruncationSelection
// ─────────────────────────────────────────────────────────────────────────────
// Only the top `fraction` of individuals are eligible to reproduce; among
// those, selection is uniform.  Common in animal/plant breeding simulations.
// ─────────────────────────────────────────────────────────────────────────────
struct TruncationSelection {
    double fraction = 0.5;  // top 50% reproduce

    [[nodiscard]] std::size_t
    operator()(const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        assert(!fitnesses.empty() && fraction > 0.0 && fraction <= 1.0);
        const std::size_t N = fitnesses.size();
        const std::size_t cutoff = std::max(
            std::size_t{1},
            static_cast<std::size_t>(std::ceil(N * fraction)));

        // Build sorted index (descending fitness).
        // For repeated use, this should be cached per generation.
        std::vector<std::size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + cutoff, order.end(),
            [&](std::size_t a, std::size_t b) {
                return fitnesses[a] > fitnesses[b];
            });

        std::uniform_int_distribution<std::size_t> pick(0, cutoff - 1);
        return order[pick(rng)];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RankSelection
// ─────────────────────────────────────────────────────────────────────────────
// Individuals are ranked by fitness; selection probability is a linear
// function of rank: P(rank r) ∝ 2 - sp + 2(sp - 1)(r - 1)/(N - 1)
// where sp ∈ [1, 2] is the "selection pressure" parameter.
//
// sp = 1 → uniform selection   sp = 2 → best individual gets 2× average
// ─────────────────────────────────────────────────────────────────────────────
struct RankSelection {
    double selection_pressure = 1.5;

    [[nodiscard]] std::size_t
    operator()(const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        assert(!fitnesses.empty());
        const std::size_t N = fitnesses.size();

        // Rank: order[0] = worst, order[N-1] = best.
        std::vector<std::size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
                return fitnesses[a] < fitnesses[b];
            });

        // Compute linear ranking weights.
        std::vector<double> weights(N);
        const double sp = std::clamp(selection_pressure, 1.0, 2.0);
        for (std::size_t i = 0; i < N; ++i) {
            double rank_frac = (N > 1)
                ? static_cast<double>(i) / static_cast<double>(N - 1)
                : 0.0;
            weights[i] = 2.0 - sp + 2.0 * (sp - 1.0) * rank_frac;
        }

        std::discrete_distribution<std::size_t> dist(
            weights.begin(), weights.end());
        return order[dist(rng)];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// BoltzmannSelection
// ─────────────────────────────────────────────────────────────────────────────
// Selection probability ∝ exp(fitness / temperature).
//
// High temperature → nearly uniform (weak selection).
// Low temperature  → strongly favours the fittest (hard selection).
//
// Useful for simulating fluctuating environments by varying T over time.
// ─────────────────────────────────────────────────────────────────────────────
struct BoltzmannSelection {
    double temperature = 1.0;

    [[nodiscard]] std::size_t
    operator()(const std::vector<Fitness>& fitnesses,
               std::mt19937_64& rng) const
    {
        assert(!fitnesses.empty() && temperature > 0.0);
        const std::size_t N = fitnesses.size();

        // Subtract max for numerical stability (log-sum-exp trick).
        double max_f = *std::max_element(fitnesses.begin(), fitnesses.end());
        std::vector<double> weights(N);
        for (std::size_t i = 0; i < N; ++i) {
            weights[i] = std::exp((fitnesses[i] - max_f) / temperature);
        }

        std::discrete_distribution<std::size_t> dist(
            weights.begin(), weights.end());
        return dist(rng);
    }
};

}  // namespace gensim
