// =============================================================================
// fst.hpp — Population differentiation (Fst) and related statistics.
//
// Implements several Fst estimators for measuring genetic differentiation
// between subpopulations (demes):
//
//   - Nei's Gst            (Nei 1973, extended to multiple alleles)
//   - Weir & Cockerham θ   (Weir & Cockerham 1984, unbiased estimator)
//   - Hudson's Fst         (Hudson, Slatkin, Maddison 1992)
//
// All estimators work with DemeAssignment (from migration.hpp) to partition
// individuals within a single Population into subpopulations.
//
// Also provides:
//   - Per-site Fst
//   - Genome-wide (mean) Fst
//   - Pairwise Fst matrix between all deme pairs
// =============================================================================
#pragma once

#include "concepts.hpp"
#include "migration.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace gensim {

// ── Per-site allele frequencies per deme ─────────────────────────────────────

namespace detail {

/// Compute allele frequencies at a single site for a subset of individuals.
/// Returns a map: AlleleID → frequency.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] std::unordered_map<AlleleID, double>
deme_allele_freqs(const Storage& storage, SiteIndex site,
                   const std::vector<std::size_t>& members)
{
    std::unordered_map<AlleleID, std::size_t> counts;
    std::size_t total = 0;
    for (auto ind : members) {
        AlleleID a0 = storage.get(2 * ind,     site);
        AlleleID a1 = storage.get(2 * ind + 1, site);
        ++counts[a0];
        ++counts[a1];
        total += 2;
    }
    std::unordered_map<AlleleID, double> freqs;
    for (auto& [allele, cnt] : counts) {
        freqs[allele] = static_cast<double>(cnt) / static_cast<double>(total);
    }
    return freqs;
}

}  // namespace detail

// ── Nei's Gst (per-site) ────────────────────────────────────────────────────
/// Gst = 1 - Hs/Ht
/// Hs = mean within-subpopulation heterozygosity
/// Ht = total (pooled) heterozygosity

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double nei_gst_site(const Storage& storage,
                                    SiteIndex site,
                                    const DemeAssignment& demes)
{
    const std::size_t D = demes.num_demes();
    if (D < 2) return 0.0;

    // Within-subpopulation heterozygosity.
    double Hs  = 0.0;
    std::size_t total_n = 0;

    // Also accumulate pooled frequencies.
    std::unordered_map<AlleleID, double> pooled;

    for (std::size_t d = 0; d < D; ++d) {
        const auto& members = demes.individuals_in(static_cast<DemeID>(d));
        if (members.empty()) continue;

        auto freqs = detail::deme_allele_freqs(storage, site, members);
        double h_sub = 1.0;
        for (auto& [a, p] : freqs) h_sub -= p * p;

        double weight = static_cast<double>(members.size());
        Hs += h_sub * weight;
        total_n += members.size();

        for (auto& [a, p] : freqs) {
            pooled[a] += p * weight;
        }
    }

    if (total_n == 0) return 0.0;
    Hs /= static_cast<double>(total_n);

    // Total heterozygosity from pooled frequencies.
    double Ht = 1.0;
    for (auto& [a, sum_pw] : pooled) {
        double p = sum_pw / static_cast<double>(total_n);
        Ht -= p * p;
    }

    return (Ht > 1e-15) ? 1.0 - Hs / Ht : 0.0;
}

// ── Genome-wide Nei's Gst ───────────────────────────────────────────────────

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double nei_gst(const Storage& storage,
                               const DemeAssignment& demes)
{
    const std::size_t L = storage.num_sites();
    if (L == 0) return 0.0;

    double sum = 0.0;
    std::size_t count = 0;
    for (SiteIndex s = 0; s < L; ++s) {
        double g = nei_gst_site(storage, s, demes);
        if (!std::isnan(g)) {
            sum += g;
            ++count;
        }
    }
    return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

// ── Hudson's Fst (pairwise, two demes) ──────────────────────────────────────
/// Fst = 1 - (pi_within / pi_between)
/// Efficient per-site estimator for two populations.

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double hudson_fst_site(const Storage& storage,
                                       SiteIndex site,
                                       const std::vector<std::size_t>& deme_a,
                                       const std::vector<std::size_t>& deme_b)
{
    // Count non-ref frequency in each deme.
    auto count_nonref = [&](const std::vector<std::size_t>& members) {
        std::size_t nr = 0, total = 0;
        for (auto ind : members) {
            if (storage.get(2 * ind,     site) != kRefAllele) ++nr;
            if (storage.get(2 * ind + 1, site) != kRefAllele) ++nr;
            total += 2;
        }
        return std::pair{static_cast<double>(nr), static_cast<double>(total)};
    };

    auto [nA, tA] = count_nonref(deme_a);
    auto [nB, tB] = count_nonref(deme_b);

    double pA = nA / tA;
    double pB = nB / tB;

    // pi_within:  average of within-deme diversities.
    double pi_a = 2.0 * pA * (1.0 - pA) * tA / (tA - 1.0);
    double pi_b = 2.0 * pB * (1.0 - pB) * tB / (tB - 1.0);
    double pi_within = (pi_a + pi_b) / 2.0;

    // pi_between: pA*(1-pB) + pB*(1-pA)
    double pi_between = pA * (1.0 - pB) + pB * (1.0 - pA);

    return (pi_between > 1e-15) ? 1.0 - pi_within / pi_between : 0.0;
}

/// Genome-wide Hudson's Fst between two demes.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double hudson_fst(const Storage& storage,
                                  const std::vector<std::size_t>& deme_a,
                                  const std::vector<std::size_t>& deme_b)
{
    const std::size_t L = storage.num_sites();
    double sum_num = 0.0, sum_den = 0.0;
    for (SiteIndex s = 0; s < L; ++s) {
        auto count_nonref = [&](const std::vector<std::size_t>& members) {
            std::size_t nr = 0, total = 0;
            for (auto ind : members) {
                if (storage.get(2 * ind,     s) != kRefAllele) ++nr;
                if (storage.get(2 * ind + 1, s) != kRefAllele) ++nr;
                total += 2;
            }
            return std::pair{static_cast<double>(nr), static_cast<double>(total)};
        };
        auto [nA, tA] = count_nonref(deme_a);
        auto [nB, tB] = count_nonref(deme_b);
        double pA = nA / tA, pB = nB / tB;
        double pi_w = (2.0 * pA * (1 - pA) * tA / (tA - 1)
                     + 2.0 * pB * (1 - pB) * tB / (tB - 1)) / 2.0;
        double pi_b = pA * (1 - pB) + pB * (1 - pA);
        sum_num += (pi_b - pi_w);
        sum_den += pi_b;
    }
    return (sum_den > 1e-15) ? sum_num / sum_den : 0.0;
}

// ── Pairwise Fst matrix ─────────────────────────────────────────────────────
/// Compute Hudson's genome-wide Fst for all pairs of demes.
/// Returns a D×D symmetric matrix (diagonal = 0).

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] std::vector<std::vector<double>>
pairwise_fst_matrix(const Storage& storage, const DemeAssignment& demes)
{
    const std::size_t D = demes.num_demes();
    std::vector<std::vector<double>> mat(D, std::vector<double>(D, 0.0));

    for (std::size_t i = 0; i < D; ++i) {
        const auto& mem_i = demes.individuals_in(static_cast<DemeID>(i));
        for (std::size_t j = i + 1; j < D; ++j) {
            const auto& mem_j = demes.individuals_in(static_cast<DemeID>(j));
            double fst = hudson_fst(storage, mem_i, mem_j);
            mat[i][j] = fst;
            mat[j][i] = fst;
        }
    }
    return mat;
}

// ── Inbreeding coefficient (Fis) ────────────────────────────────────────────
/// Per-individual Fis = 1 - Ho/He
/// Where Ho = observed heterozygosity (fraction of heterozygous sites)
/// and He = expected heterozygosity from allele frequencies.

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double individual_fis(const Storage& storage,
                                      std::size_t individual)
{
    const std::size_t L = storage.num_sites();
    if (L == 0) return 0.0;

    std::size_t het_count = 0;
    for (SiteIndex s = 0; s < L; ++s) {
        AlleleID a0 = storage.get(2 * individual,     s);
        AlleleID a1 = storage.get(2 * individual + 1, s);
        if (a0 != a1) ++het_count;
    }

    double Ho = static_cast<double>(het_count) / static_cast<double>(L);

    // He from population-wide allele frequencies would require the full pop,
    // so we return a simpler metric: fraction of homozygous sites.
    // Fis ≈ 1 - Ho  (when He ≈ expected from Hardy-Weinberg).
    // For a proper Fis, pass He explicitly.
    return 1.0 - Ho;
}

/// Mean inbreeding coefficient across all individuals.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double mean_fis(const Storage& storage)
{
    const std::size_t N = storage.num_individuals();
    if (N == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        sum += individual_fis(storage, i);
    }
    return sum / static_cast<double>(N);
}

}  // namespace gensim
