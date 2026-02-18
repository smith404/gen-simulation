// =============================================================================
// quantitative_trait.hpp — Quantitative trait models and phenotype calculation.
//
// Maps genotype → phenotype for one or more quantitative traits.  Fitness
// can then be a function of phenotype rather than directly of genotype,
// enabling much richer ecological models.
//
// The fundamental model:
//   Phenotype P = Genetic value G + Environmental noise E
//   G = f(genotype)   — additive, dominance, epistatic components
//   E ~ N(0, σ²_E)   — environmental variance
//
// Provided trait models:
//
//   AdditiveTraitModel        – G = Σ effect sizes (sel_coeff)
//   DominanceTraitModel       – G = Σ (additive + dominance deviations)
//   MultiTraitAdditiveModel   – multiple traits with pleiotropy
//   QuantitativeTraitPolicy   – combines any trait model + env noise + fitness fn
//
// Statistics:
//   compute_trait_statistics  – V_A, V_P, h², mean, variance
//
// Integration:
//   The trait model is evaluated each generation via
//   Population::compute_traits() which fills Individual::traits.  Then
//   a trait-based fitness function uses those trait values.
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace gensim {

// ── Trait statistics ────────────────────────────────────────────────────────
struct TraitStatistics {
    double mean          = 0.0;   // mean phenotype
    double variance      = 0.0;   // phenotypic variance V_P
    double genetic_var   = 0.0;   // additive genetic variance V_A (approx)
    double env_var       = 0.0;   // environmental variance V_E
    double heritability  = 0.0;   // narrow-sense h² = V_A / V_P
    double min_val       = 0.0;
    double max_val       = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// AdditiveTraitModel — single trait, purely additive.
// ─────────────────────────────────────────────────────────────────────────────
// Genetic value G = Σ_sites (effect_0 + effect_1) where effects come from
// AlleleInfo::sel_coeff.  (The sel_coeff field doubles as the effect size
// for quantitative traits.)
struct AdditiveTraitModel {
    double env_variance = 0.0;   // σ²_E for environmental noise

    /// Compute genetic value for one individual (Dense backend).
    [[nodiscard]] double genetic_value(const DenseHaplotypes& storage,
                                        std::size_t ind,
                                        const AlleleTable& table) const noexcept
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        double g = 0.0;
        for (std::size_t s = 0; s < L; ++s) {
            if (h0[s] != kRefAllele) g += table[h0[s]].sel_coeff;
            if (h1[s] != kRefAllele) g += table[h1[s]].sel_coeff;
        }
        return g;
    }

    /// Compute genetic value (Sparse backend).
    [[nodiscard]] double genetic_value(const SparseVariants& storage,
                                        std::size_t ind,
                                        const AlleleTable& table) const noexcept
    {
        double g = 0.0;
        for (auto& v : storage.variants()) {
            AlleleID a0 = v.alleles[2 * ind];
            AlleleID a1 = v.alleles[2 * ind + 1];
            if (a0 != kRefAllele) g += table[a0].sel_coeff;
            if (a1 != kRefAllele) g += table[a1].sel_coeff;
        }
        return g;
    }

    /// Compute phenotype = G + E.
    template <typename Storage>
    [[nodiscard]] double phenotype(const Storage& storage,
                                    std::size_t ind,
                                    const AlleleTable& table,
                                    std::mt19937_64& rng) const
    {
        double g = genetic_value(storage, ind, table);
        if (env_variance > 0.0) {
            std::normal_distribution<double> noise(0.0, std::sqrt(env_variance));
            g += noise(rng);
        }
        return g;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DominanceTraitModel — single trait with dominance deviations.
// ─────────────────────────────────────────────────────────────────────────────
// Per-site contribution depends on genotype:
//   - Homozygous ref:     d = 0
//   - Heterozygous:       d = h_i * a_i      (h = dominance, a = effect)
//   - Homozygous derived: d = a_i             (full effect)
//   - Compound het:       d = h * a_i + h * a_j
//
// G = Σ_sites d_i
struct DominanceTraitModel {
    double env_variance = 0.0;

    [[nodiscard]] double genetic_value(const DenseHaplotypes& storage,
                                        std::size_t ind,
                                        const AlleleTable& table) const noexcept
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        double g = 0.0;
        for (std::size_t s = 0; s < L; ++s) {
            const bool alt0 = (h0[s] != kRefAllele);
            const bool alt1 = (h1[s] != kRefAllele);
            if (!alt0 && !alt1) continue;

            if (alt0 && alt1 && h0[s] == h1[s]) {
                g += table[h0[s]].sel_coeff;
            } else if (alt0 && alt1) {
                g += table[h0[s]].dominance_h * table[h0[s]].sel_coeff;
                g += table[h1[s]].dominance_h * table[h1[s]].sel_coeff;
            } else {
                AlleleID a = alt0 ? h0[s] : h1[s];
                g += table[a].dominance_h * table[a].sel_coeff;
            }
        }
        return g;
    }

    template <typename Storage>
    [[nodiscard]] double phenotype(const Storage& storage,
                                    std::size_t ind,
                                    const AlleleTable& table,
                                    std::mt19937_64& rng) const
    {
        double g = genetic_value(storage, ind, table);
        if (env_variance > 0.0) {
            std::normal_distribution<double> noise(0.0, std::sqrt(env_variance));
            g += noise(rng);
        }
        return g;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiTraitAdditiveModel — multiple quantitative traits with pleiotropy.
// ─────────────────────────────────────────────────────────────────────────────
// Each allele can affect multiple traits.  The effect on trait `t` is stored
// in an effect vector associated with each AlleleID.
//
// Since AlleleInfo only has one sel_coeff, we use an external per-allele
// effect matrix for multi-trait models.
struct MultiTraitAdditiveModel {
    std::size_t num_traits = 1;
    std::vector<double> env_variances;   // one per trait

    // ── Effect storage ──────────────────────────────────────────────────────
    // effects[allele_id][trait_index] = effect size of that allele on trait t.
    // For alleles not in this map, the effect is 0 (reference allele).
    std::vector<std::vector<double>> effects;

    /// Register an allele's effects on all traits.
    void set_effects(AlleleID id, std::vector<double> trait_effects) {
        if (id >= effects.size()) effects.resize(id + 1);
        effects[id] = std::move(trait_effects);
    }

    /// Get the effect of an allele on a specific trait.
    [[nodiscard]] double get_effect(AlleleID id, std::size_t trait) const noexcept {
        if (id == kRefAllele) return 0.0;
        if (id >= effects.size() || trait >= effects[id].size()) return 0.0;
        return effects[id][trait];
    }

    /// Compute genetic values for all traits.
    [[nodiscard]] std::vector<double>
    genetic_values(const DenseHaplotypes& storage,
                   std::size_t ind) const
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        std::vector<double> g(num_traits, 0.0);
        for (std::size_t s = 0; s < L; ++s) {
            if (h0[s] != kRefAllele) {
                for (std::size_t t = 0; t < num_traits; ++t)
                    g[t] += get_effect(h0[s], t);
            }
            if (h1[s] != kRefAllele) {
                for (std::size_t t = 0; t < num_traits; ++t)
                    g[t] += get_effect(h1[s], t);
            }
        }
        return g;
    }

    /// Compute phenotypes = G + E for all traits.
    [[nodiscard]] std::vector<double>
    phenotypes(const DenseHaplotypes& storage,
               std::size_t ind,
               std::mt19937_64& rng) const
    {
        auto g = genetic_values(storage, ind);
        for (std::size_t t = 0; t < num_traits; ++t) {
            double ve = (t < env_variances.size()) ? env_variances[t] : 0.0;
            if (ve > 0.0) {
                std::normal_distribution<double> noise(0.0, std::sqrt(ve));
                g[t] += noise(rng);
            }
        }
        return g;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Trait-based fitness functions.
// ─────────────────────────────────────────────────────────────────────────────

/// Stabilizing selection on a single trait: w = exp(-(z - θ)² / (2V_s))
struct StabilizingTraitFitness {
    std::size_t trait_index = 0;
    double optimum = 0.0;     // θ
    double Vs = 1.0;          // strength of selection

    [[nodiscard]] Fitness operator()(const std::vector<Individual>& inds,
                                     std::size_t ind) const noexcept;
};

/// Directional selection: w = 1 + β * z  (truncated at 0).
struct DirectionalTraitFitness {
    std::size_t trait_index = 0;
    double beta = 0.01;       // selection gradient

    [[nodiscard]] Fitness operator()(const std::vector<Individual>& inds,
                                     std::size_t ind) const noexcept;
};

/// Correlational selection on two traits:
///   w = exp(-[(z1 - θ1)² + (z2 - θ2)² + 2ρ(z1-θ1)(z2-θ2)] / (2V_s))
struct CorrelationalTraitFitness {
    std::size_t trait_1 = 0, trait_2 = 1;
    double optimum_1 = 0.0, optimum_2 = 0.0;
    double Vs = 1.0;
    double rho = 0.0;  // correlation between trait optima

    [[nodiscard]] Fitness operator()(const std::vector<Individual>& inds,
                                     std::size_t ind) const noexcept;
};

// ── Inline definitions for trait-based fitness ──────────────────────────────

// We need Individual to be fully defined.  Since Individual is in population.hpp
// and this header is meant to be standalone, we use a forward-declared version.
// Users include population.hpp which defines Individual before using these.

inline Fitness StabilizingTraitFitness::operator()(
    const std::vector<Individual>& inds, std::size_t ind) const noexcept
{
    if (trait_index >= inds[ind].traits.size()) return 1.0;
    double z = inds[ind].traits[trait_index];
    double diff = z - optimum;
    return std::exp(-(diff * diff) / (2.0 * Vs));
}

inline Fitness DirectionalTraitFitness::operator()(
    const std::vector<Individual>& inds, std::size_t ind) const noexcept
{
    if (trait_index >= inds[ind].traits.size()) return 1.0;
    double z = inds[ind].traits[trait_index];
    return std::max(1.0 + beta * z, 0.0);
}

inline Fitness CorrelationalTraitFitness::operator()(
    const std::vector<Individual>& inds, std::size_t ind) const noexcept
{
    double z1 = (trait_1 < inds[ind].traits.size()) ? inds[ind].traits[trait_1] : 0.0;
    double z2 = (trait_2 < inds[ind].traits.size()) ? inds[ind].traits[trait_2] : 0.0;
    double d1 = z1 - optimum_1;
    double d2 = z2 - optimum_2;
    double exponent = -(d1 * d1 + d2 * d2 + 2.0 * rho * d1 * d2) / (2.0 * Vs);
    return std::exp(exponent);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fluctuating (moving) optimum — optimum shifts over time.
// ─────────────────────────────────────────────────────────────────────────────
// The optimum moves linearly or sinusoidally over generations, modelling
// environmental change.
struct MovingOptimumSchedule {
    double initial_optimum = 0.0;
    double rate = 0.0;           // linear shift per generation (0 = static)
    double amplitude = 0.0;      // sinusoidal amplitude (0 = no oscillation)
    double period = 100.0;       // oscillation period in generations

    [[nodiscard]] double optimum_at(Generation g) const noexcept {
        double opt = initial_optimum + rate * static_cast<double>(g);
        if (amplitude != 0.0) {
            opt += amplitude * std::sin(2.0 * 3.14159265358979323846 *
                                         static_cast<double>(g) / period);
        }
        return opt;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Frequency-dependent fitness — fitness of a genotype depends on its
// frequency in the population.
// ─────────────────────────────────────────────────────────────────────────────
// Negative frequency-dependent selection (rare-allele advantage) is modelled
// as: w_i = base_fitness * (1 + strength * (1 - freq_i))
// where freq_i is the frequency of individual i's phenotypic class.
//
// This is important for self-incompatibility loci, host-parasite arms races,
// and the maintenance of polymorphism.
struct FrequencyDependentFitness {
    double strength = 0.1;       // > 0 = rare advantage, < 0 = common advantage
    std::size_t trait_index = 0; // trait used to define phenotypic classes
    double bin_width = 0.1;      // bin size for discretising continuous traits

    /// Compute frequency-dependent fitnesses for all individuals.
    /// Returns a vector of fitness adjustments (multiply with base fitness).
    [[nodiscard]] std::vector<double>
    compute_adjustments(const std::vector<Individual>& inds) const
    {
        if (inds.empty()) return {};

        // Bin individuals by their trait value.
        std::unordered_map<std::int64_t, std::size_t> bin_counts;
        std::vector<std::int64_t> ind_bin(inds.size());

        for (std::size_t i = 0; i < inds.size(); ++i) {
            double z = (trait_index < inds[i].traits.size())
                           ? inds[i].traits[trait_index] : 0.0;
            ind_bin[i] = static_cast<std::int64_t>(std::floor(z / bin_width));
            bin_counts[ind_bin[i]]++;
        }

        double N = static_cast<double>(inds.size());
        std::vector<double> adjustments(inds.size());
        for (std::size_t i = 0; i < inds.size(); ++i) {
            double freq = static_cast<double>(bin_counts[ind_bin[i]]) / N;
            adjustments[i] = 1.0 + strength * (1.0 - freq);
        }
        return adjustments;
    }

private:
    // Helper to avoid including <unordered_map> in the concept forward decl.
    // (It's already included above.)
    using map_type = std::unordered_map<std::int64_t, std::size_t>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Compute summary statistics for quantitative traits.
// ─────────────────────────────────────────────────────────────────────────────
inline TraitStatistics
compute_trait_statistics(const std::vector<Individual>& inds,
                          std::size_t trait_index,
                          double env_variance = 0.0)
{
    TraitStatistics ts;
    if (inds.empty()) return ts;

    std::vector<double> vals;
    vals.reserve(inds.size());
    for (auto& ind : inds) {
        if (trait_index < ind.traits.size())
            vals.push_back(ind.traits[trait_index]);
    }
    if (vals.empty()) return ts;

    // Mean.
    ts.mean = std::accumulate(vals.begin(), vals.end(), 0.0) /
              static_cast<double>(vals.size());

    // Variance.
    double ss = 0.0;
    for (double v : vals) {
        double d = v - ts.mean;
        ss += d * d;
    }
    ts.variance = ss / static_cast<double>(vals.size());

    ts.min_val = *std::min_element(vals.begin(), vals.end());
    ts.max_val = *std::max_element(vals.begin(), vals.end());

    // V_P = V_A + V_E.  We assume V_E is known (env_variance parameter).
    ts.env_var = env_variance;
    ts.genetic_var = std::max(ts.variance - env_variance, 0.0);
    ts.heritability = (ts.variance > 0.0)
                          ? ts.genetic_var / ts.variance
                          : 0.0;

    return ts;
}

}  // namespace gensim
