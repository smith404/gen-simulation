// =============================================================================
// fitness_model.hpp — Pluggable fitness policies.
//
// Provided models:
//
//   1. NeutralFitness        – every individual has fitness 1.0.
//   2. AdditiveFitness       – fitness = 1 + Σ_sites Σ_copies s_i, where s_i
//                              is the selection coefficient stored in AlleleTable.
//   3. MultiplicativeFitness – fitness = Π_sites (1 + s_i * count_i).
//
// Each model must provide:
//   Fitness operator()(const Storage& storage, size_t individual,
//                      const AlleleTable& table) const;
//
// Extension: wrap any callable with the above signature as a fitness policy.
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>

namespace gensim {

// ─────────────────────────────────────────────────────────────────────────────
// NeutralFitness
// ─────────────────────────────────────────────────────────────────────────────
struct NeutralFitness {
    // Dense
    [[nodiscard]] Fitness operator()(const DenseHaplotypes& /*storage*/,
                                     std::size_t /*individual*/,
                                     const AlleleTable& /*table*/) const noexcept {
        return 1.0;
    }
    // Sparse
    [[nodiscard]] Fitness operator()(const SparseVariants& /*storage*/,
                                     std::size_t /*individual*/,
                                     const AlleleTable& /*table*/) const noexcept {
        return 1.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AdditiveFitness  (dominance-aware)
// ─────────────────────────────────────────────────────────────────────────────
// Fitness per site depends on genotype:
//   - Homozygous reference (0/0): contribution = 0
//   - Heterozygous (one non-ref):  contribution = h * s
//   - Homozygous derived (both non-ref, same allele): contribution = s
//   - Compound heterozygote (both non-ref, different): h*s1 + h*s2
// Total: w = 1 + sum of contributions.  Clamped to [0, inf).
// ─────────────────────────────────────────────────────────────────────────────
struct AdditiveFitness {
    // Dense
    [[nodiscard]] Fitness operator()(const DenseHaplotypes& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        double w = 1.0;
        for (std::size_t s = 0; s < L; ++s) {
            const bool alt0 = (h0[s] != kRefAllele);
            const bool alt1 = (h1[s] != kRefAllele);
            if (!alt0 && !alt1) continue;   // homozygous ref -- no effect

            if (alt0 && alt1 && h0[s] == h1[s]) {
                // Homozygous derived: full effect s
                w += table[h0[s]].sel_coeff;
            } else if (alt0 && alt1) {
                // Compound heterozygote: h*s for each allele
                w += table[h0[s]].dominance_h * table[h0[s]].sel_coeff;
                w += table[h1[s]].dominance_h * table[h1[s]].sel_coeff;
            } else {
                // Simple heterozygote: h*s
                AlleleID a = alt0 ? h0[s] : h1[s];
                w += table[a].dominance_h * table[a].sel_coeff;
            }
        }
        return std::max(w, 0.0);
    }

    // Sparse
    [[nodiscard]] Fitness operator()(const SparseVariants& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        double w = 1.0;
        for (auto& v : storage.variants()) {
            AlleleID a0 = v.alleles[2 * ind];
            AlleleID a1 = v.alleles[2 * ind + 1];
            const bool alt0 = (a0 != kRefAllele);
            const bool alt1 = (a1 != kRefAllele);
            if (!alt0 && !alt1) continue;

            if (alt0 && alt1 && a0 == a1) {
                w += table[a0].sel_coeff;
            } else if (alt0 && alt1) {
                w += table[a0].dominance_h * table[a0].sel_coeff;
                w += table[a1].dominance_h * table[a1].sel_coeff;
            } else {
                AlleleID a = alt0 ? a0 : a1;
                w += table[a].dominance_h * table[a].sel_coeff;
            }
        }
        return std::max(w, 0.0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiplicativeFitness  (dominance-aware)
// ─────────────────────────────────────────────────────────────────────────────
// Per-site multiplicative with dominance:
//   - Heterozygote: w *= (1 + h*s)
//   - Homozygous derived: w *= (1 + s)
// ─────────────────────────────────────────────────────────────────────────────
struct MultiplicativeFitness {
    // Dense
    [[nodiscard]] Fitness operator()(const DenseHaplotypes& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        double w = 1.0;
        for (std::size_t s = 0; s < L; ++s) {
            const bool alt0 = (h0[s] != kRefAllele);
            const bool alt1 = (h1[s] != kRefAllele);
            if (!alt0 && !alt1) continue;

            if (alt0 && alt1 && h0[s] == h1[s]) {
                w *= (1.0 + table[h0[s]].sel_coeff);
            } else if (alt0 && alt1) {
                w *= (1.0 + table[h0[s]].dominance_h * table[h0[s]].sel_coeff);
                w *= (1.0 + table[h1[s]].dominance_h * table[h1[s]].sel_coeff);
            } else {
                AlleleID a = alt0 ? h0[s] : h1[s];
                w *= (1.0 + table[a].dominance_h * table[a].sel_coeff);
            }
        }
        return std::max(w, 0.0);
    }

    // Sparse
    [[nodiscard]] Fitness operator()(const SparseVariants& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        double w = 1.0;
        for (auto& v : storage.variants()) {
            AlleleID a0 = v.alleles[2 * ind];
            AlleleID a1 = v.alleles[2 * ind + 1];
            const bool alt0 = (a0 != kRefAllele);
            const bool alt1 = (a1 != kRefAllele);
            if (!alt0 && !alt1) continue;

            if (alt0 && alt1 && a0 == a1) {
                w *= (1.0 + table[a0].sel_coeff);
            } else if (alt0 && alt1) {
                w *= (1.0 + table[a0].dominance_h * table[a0].sel_coeff);
                w *= (1.0 + table[a1].dominance_h * table[a1].sel_coeff);
            } else {
                AlleleID a = alt0 ? a0 : a1;
                w *= (1.0 + table[a].dominance_h * table[a].sel_coeff);
            }
        }
        return std::max(w, 0.0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// StabilizingSelectionFitness
// ─────────────────────────────────────────────────────────────────────────────
// Quantitative trait: z = sum of allele effect sizes across both haplotypes.
// Fitness: w = exp(-(z - optimum)^2 / (2 * Vs))
// Uses sel_coeff as the "effect size" of each allele.
// ─────────────────────────────────────────────────────────────────────────────
struct StabilizingSelectionFitness {
    double optimum = 0.0;     // optimal trait value (theta)
    double Vs      = 1.0;     // strength of stabilizing selection (variance)

    // Dense
    [[nodiscard]] Fitness operator()(const DenseHaplotypes& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        const std::size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        double z = 0.0;
        for (std::size_t s = 0; s < L; ++s) {
            if (h0[s] != kRefAllele) z += table[h0[s]].sel_coeff;
            if (h1[s] != kRefAllele) z += table[h1[s]].sel_coeff;
        }
        double diff = z - optimum;
        return std::exp(-(diff * diff) / (2.0 * Vs));
    }

    // Sparse
    [[nodiscard]] Fitness operator()(const SparseVariants& storage,
                                     std::size_t ind,
                                     const AlleleTable& table) const noexcept
    {
        double z = 0.0;
        for (auto& v : storage.variants()) {
            AlleleID a0 = v.alleles[2 * ind];
            AlleleID a1 = v.alleles[2 * ind + 1];
            if (a0 != kRefAllele) z += table[a0].sel_coeff;
            if (a1 != kRefAllele) z += table[a1].sel_coeff;
        }
        double diff = z - optimum;
        return std::exp(-(diff * diff) / (2.0 * Vs));
    }
};

}  // namespace gensim
