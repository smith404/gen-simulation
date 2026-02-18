// =============================================================================
// statistics.hpp — Population genetics summary statistics.
//
// Provides summary statistics that work on any storage backend.
//
// Original backend-specific functions (compute_sfs_dense, compute_sfs_sparse,
// etc.) are retained for backward compatibility.  New concept-constrained
// template functions (compute_sfs, compute_pi, compute_summary, etc.) auto-
// dispatch to the best implementation for any conforming storage type.
//
// Implemented statistics:
//   - Site Frequency Spectrum (SFS), folded and unfolded
//   - Nucleotide diversity (pi)
//   - Watterson's theta (theta_W)
//   - Tajima's D
//   - Per-site heterozygosity
//   - Mean heterozygosity across all sites
//   - Allele frequency at a given site for a given allele
// =============================================================================
#pragma once

#include "concepts.hpp"
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace gensim {

// ── Harmonic number helpers ─────────────────────────────────────────────────
// a1 = sum_{i=1}^{n-1} 1/i,   a2 = sum_{i=1}^{n-1} 1/i^2
// Used by Watterson's theta and Tajima's D.

inline double harmonic_a1(std::size_t n) {
    double a = 0.0;
    for (std::size_t i = 1; i < n; ++i) a += 1.0 / static_cast<double>(i);
    return a;
}

inline double harmonic_a2(std::size_t n) {
    double a = 0.0;
    for (std::size_t i = 1; i < n; ++i)
        a += 1.0 / (static_cast<double>(i) * static_cast<double>(i));
    return a;
}

// =============================================================================
// Site Frequency Spectrum (SFS)
// =============================================================================

/// Compute the unfolded SFS.  Returns a vector of size 2N+1 where sfs[k] is
/// the number of sites with exactly k copies of a non-reference allele.
/// Sites that are monomorphic for the reference (k=0) are placed in sfs[0].
///
/// For the infinite-alleles model, each site has at most one non-ref allele,
/// so k is simply the count of that allele.  If multiple non-ref alleles exist
/// at a site (finite-alleles), k = total non-ref count.
inline std::vector<std::size_t>
compute_sfs_dense(const DenseHaplotypes& storage) {
    const std::size_t H = storage.num_haplotypes();
    const std::size_t L = storage.num_sites();
    std::vector<std::size_t> sfs(H + 1, 0);

    for (std::size_t s = 0; s < L; ++s) {
        std::size_t non_ref = 0;
        for (std::size_t h = 0; h < H; ++h) {
            if (storage.hap_ptr(h)[s] != kRefAllele) ++non_ref;
        }
        ++sfs[non_ref];
    }
    return sfs;
}

inline std::vector<std::size_t>
compute_sfs_sparse(const SparseVariants& storage) {
    const std::size_t H = storage.num_haplotypes();
    const std::size_t L = storage.num_sites();

    std::vector<std::size_t> sfs(H + 1, 0);

    // Count from variant records.
    std::size_t variant_sites = 0;
    for (auto& v : storage.variants()) {
        std::size_t non_ref = 0;
        for (auto a : v.alleles) {
            if (a != kRefAllele) ++non_ref;
        }
        ++sfs[non_ref];
        ++variant_sites;
    }
    // All non-variant sites are monomorphic reference.
    sfs[0] += (L - variant_sites);
    return sfs;
}

/// Fold the SFS: combine counts k and 2N-k into the minor allele frequency
/// spectrum.  Returns vector of size N+1 (indices 0..N).
inline std::vector<std::size_t>
fold_sfs(const std::vector<std::size_t>& sfs) {
    const std::size_t H = sfs.size() - 1;   // 2N
    const std::size_t half = H / 2;
    std::vector<std::size_t> folded(half + 1, 0);
    for (std::size_t k = 0; k <= half; ++k) {
        folded[k] = sfs[k];
        if (k != H - k) folded[k] += sfs[H - k];
    }
    return folded;
}

// =============================================================================
// Nucleotide diversity (pi)
// =============================================================================
// pi = (1 / C(n,2)) * sum_sites sum_{i<j} (x_i != x_j)
//    = (1 / C(n,2)) * sum_sites k*(n-k) / 1  (for biallelic)
//    where n = 2N (number of haplotypes), k = non-ref count.
//
// For multi-allelic sites, the pairwise difference contribution at a site is:
//   sum_{alleles a} freq_a * (1 - freq_a) * n / (n-1) (per-site heterozygosity)
// We use the simpler biallelic formula (ref vs non-ref).

inline double compute_pi_from_sfs(const std::vector<std::size_t>& sfs) {
    const std::size_t n = sfs.size() - 1;  // number of haplotypes (2N)
    if (n < 2) return 0.0;
    const double denom = static_cast<double>(n) * static_cast<double>(n - 1) / 2.0;

    double pi = 0.0;
    for (std::size_t k = 1; k < n; ++k) {  // skip k=0 (monomorphic) and k=n (fixed)
        double contrib = static_cast<double>(k) * static_cast<double>(n - k);
        pi += static_cast<double>(sfs[k]) * contrib;
    }
    return pi / denom;
}

/// Convenience: compute pi directly from storage.
inline double compute_pi_dense(const DenseHaplotypes& s) {
    return compute_pi_from_sfs(compute_sfs_dense(s));
}
inline double compute_pi_sparse(const SparseVariants& s) {
    return compute_pi_from_sfs(compute_sfs_sparse(s));
}

// =============================================================================
// Watterson's theta (theta_W)
// =============================================================================
// theta_W = S / a1,  where S = number of segregating sites,
// a1 = sum_{i=1}^{n-1} 1/i.

inline double compute_theta_w(std::size_t S, std::size_t n) {
    if (n < 2) return 0.0;
    return static_cast<double>(S) / harmonic_a1(n);
}

/// Convenience: count S from the SFS (all entries except 0 and n).
inline std::size_t segregating_sites_from_sfs(const std::vector<std::size_t>& sfs) {
    std::size_t S = 0;
    const std::size_t n = sfs.size() - 1;
    for (std::size_t k = 1; k < n; ++k) S += sfs[k];
    return S;
}

// =============================================================================
// Tajima's D
// =============================================================================
// D = (pi - theta_W) / sqrt(Var)
//
// Var = e1 * S + e2 * S * (S - 1)
// where e1, e2 are functions of n (see Tajima 1989).

inline double compute_tajimas_d(double pi, double theta_w,
                                  std::size_t S, std::size_t n)
{
    if (S == 0 || n < 2) return 0.0;

    double a1 = harmonic_a1(n);
    double a2 = harmonic_a2(n);
    double dn = static_cast<double>(n);

    double b1 = (dn + 1.0) / (3.0 * (dn - 1.0));
    double b2 = 2.0 * (dn * dn + dn + 3.0) / (9.0 * dn * (dn - 1.0));

    double c1 = b1 - 1.0 / a1;
    double c2 = b2 - (dn + 2.0) / (a1 * dn) + a2 / (a1 * a1);

    double e1 = c1 / a1;
    double e2 = c2 / (a1 * a1 + a2);

    double dS = static_cast<double>(S);
    double var = e1 * dS + e2 * dS * (dS - 1.0);
    if (var <= 0.0) return 0.0;

    return (pi - theta_w) / std::sqrt(var);
}

// =============================================================================
// Per-site and mean heterozygosity
// =============================================================================
// Per-site expected heterozygosity = 1 - sum(p_i^2) where p_i = freq of allele i.
// Computed over all 2N haplotypes.

inline double site_heterozygosity_dense(const DenseHaplotypes& storage,
                                          SiteIndex site)
{
    auto counts = storage.allele_counts(site);
    double n = static_cast<double>(storage.num_haplotypes());
    double sum_p2 = 0.0;
    for (auto& [a, c] : counts) {
        double p = static_cast<double>(c) / n;
        sum_p2 += p * p;
    }
    return 1.0 - sum_p2;
}

inline double site_heterozygosity_sparse(const SparseVariants& storage,
                                           SiteIndex site)
{
    auto counts = storage.allele_counts(site);
    double n = static_cast<double>(storage.num_haplotypes());
    double sum_p2 = 0.0;
    for (auto& [a, c] : counts) {
        double p = static_cast<double>(c) / n;
        sum_p2 += p * p;
    }
    return 1.0 - sum_p2;
}

/// Mean heterozygosity across all sites (Dense).
inline double mean_heterozygosity_dense(const DenseHaplotypes& storage) {
    const std::size_t L = storage.num_sites();
    if (L == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t s = 0; s < L; ++s) {
        sum += site_heterozygosity_dense(storage, s);
    }
    return sum / static_cast<double>(L);
}

/// Mean heterozygosity across all sites (Sparse).
/// Only variant sites contribute non-zero heterozygosity.
inline double mean_heterozygosity_sparse(const SparseVariants& storage) {
    const std::size_t L = storage.num_sites();
    if (L == 0) return 0.0;
    double sum = 0.0;
    for (auto& v : storage.variants()) {
        // Compute heterozygosity for this variant site inline.
        double n = static_cast<double>(storage.num_haplotypes());
        std::vector<std::pair<AlleleID, std::size_t>> counts;
        for (auto a : v.alleles) {
            auto it = std::find_if(counts.begin(), counts.end(),
                                   [a](auto& p){ return p.first == a; });
            if (it != counts.end()) ++it->second;
            else counts.emplace_back(a, 1);
        }
        double sum_p2 = 0.0;
        for (auto& [a, c] : counts) {
            double p = static_cast<double>(c) / n;
            sum_p2 += p * p;
        }
        sum += (1.0 - sum_p2);
    }
    return sum / static_cast<double>(L);
}

// =============================================================================
// Convenience: compute all summary stats at once
// =============================================================================

struct SummaryStats {
    std::size_t              num_haplotypes;      // n = 2N
    std::size_t              num_sites;            // L
    std::size_t              segregating_sites;    // S
    double                   pi;                   // nucleotide diversity
    double                   theta_w;              // Watterson's theta
    double                   tajimas_d;            // Tajima's D
    double                   mean_heterozygosity;  // mean H across sites
    std::vector<std::size_t> sfs;                  // unfolded SFS (size 2N+1)
};

inline SummaryStats compute_summary_dense(const DenseHaplotypes& storage) {
    SummaryStats ss;
    ss.num_haplotypes = storage.num_haplotypes();
    ss.num_sites      = storage.num_sites();
    ss.sfs            = compute_sfs_dense(storage);
    ss.segregating_sites = segregating_sites_from_sfs(ss.sfs);
    ss.pi             = compute_pi_from_sfs(ss.sfs);
    ss.theta_w        = compute_theta_w(ss.segregating_sites, ss.num_haplotypes);
    ss.tajimas_d      = compute_tajimas_d(ss.pi, ss.theta_w,
                                           ss.segregating_sites, ss.num_haplotypes);
    ss.mean_heterozygosity = mean_heterozygosity_dense(storage);
    return ss;
}

inline SummaryStats compute_summary_sparse(const SparseVariants& storage) {
    SummaryStats ss;
    ss.num_haplotypes = storage.num_haplotypes();
    ss.num_sites      = storage.num_sites();
    ss.sfs            = compute_sfs_sparse(storage);
    ss.segregating_sites = segregating_sites_from_sfs(ss.sfs);
    ss.pi             = compute_pi_from_sfs(ss.sfs);
    ss.theta_w        = compute_theta_w(ss.segregating_sites, ss.num_haplotypes);
    ss.tajimas_d      = compute_tajimas_d(ss.pi, ss.theta_w,
                                           ss.segregating_sites, ss.num_haplotypes);
    ss.mean_heterozygosity = mean_heterozygosity_sparse(storage);
    return ss;
}

/// Print a compact SFS histogram (non-zero buckets only).
inline void print_sfs(const std::vector<std::size_t>& sfs, std::size_t max_k = 20) {
    std::size_t limit = std::min(max_k, sfs.size());
    for (std::size_t k = 1; k < limit; ++k) {
        if (sfs[k] > 0) {
            std::cout << "    SFS[" << k << "]=" << sfs[k] << "\n";
        }
    }
    // Summarise the tail.
    std::size_t tail = 0;
    for (std::size_t k = limit; k < sfs.size() - 1; ++k) tail += sfs[k];
    if (tail > 0) std::cout << "    SFS[" << limit << "..]=(" << tail << " sites)\n";
}

// =============================================================================
// Generic concept-constrained overloads
// =============================================================================
// These dispatch to the optimal backend-specific implementation when available
// and fall back to element-by-element access for custom storage backends.

/// Generic SFS — uses element access for any conforming backend.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] std::vector<std::size_t>
compute_sfs(const Storage& storage) {
    if constexpr (std::same_as<Storage, DenseHaplotypes>) {
        return compute_sfs_dense(storage);
    } else if constexpr (std::same_as<Storage, SparseVariants>) {
        return compute_sfs_sparse(storage);
    } else {
        // Generic fallback via element access.
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        std::vector<std::size_t> sfs(H + 1, 0);
        for (std::size_t s = 0; s < L; ++s) {
            std::size_t non_ref = 0;
            for (std::size_t h = 0; h < H; ++h) {
                if (storage.get(h, static_cast<SiteIndex>(s)) != kRefAllele)
                    ++non_ref;
            }
            ++sfs[non_ref];
        }
        return sfs;
    }
}

/// Generic nucleotide diversity.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double compute_pi(const Storage& storage) {
    return compute_pi_from_sfs(compute_sfs(storage));
}

/// Generic site heterozygosity.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double site_heterozygosity(const Storage& storage, SiteIndex site) {
    if constexpr (std::same_as<Storage, DenseHaplotypes>) {
        return site_heterozygosity_dense(storage, site);
    } else if constexpr (std::same_as<Storage, SparseVariants>) {
        return site_heterozygosity_sparse(storage, site);
    } else {
        // Generic: count alleles via element access.
        const std::size_t H = storage.num_haplotypes();
        std::vector<std::pair<AlleleID, std::size_t>> counts;
        for (std::size_t h = 0; h < H; ++h) {
            AlleleID a = storage.get(h, site);
            auto it = std::find_if(counts.begin(), counts.end(),
                                   [a](auto& p){ return p.first == a; });
            if (it != counts.end()) ++it->second;
            else counts.emplace_back(a, 1);
        }
        double n = static_cast<double>(H);
        double sum_p2 = 0.0;
        for (auto& [a, c] : counts) {
            double p = static_cast<double>(c) / n;
            sum_p2 += p * p;
        }
        return 1.0 - sum_p2;
    }
}

/// Generic mean heterozygosity.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] double mean_heterozygosity(const Storage& storage) {
    if constexpr (std::same_as<Storage, DenseHaplotypes>) {
        return mean_heterozygosity_dense(storage);
    } else if constexpr (std::same_as<Storage, SparseVariants>) {
        return mean_heterozygosity_sparse(storage);
    } else {
        const std::size_t L = storage.num_sites();
        if (L == 0) return 0.0;
        double sum = 0.0;
        for (std::size_t s = 0; s < L; ++s) {
            sum += site_heterozygosity(storage, static_cast<SiteIndex>(s));
        }
        return sum / static_cast<double>(L);
    }
}

/// Generic summary statistics.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] SummaryStats compute_summary(const Storage& storage) {
    if constexpr (std::same_as<Storage, DenseHaplotypes>) {
        return compute_summary_dense(storage);
    } else if constexpr (std::same_as<Storage, SparseVariants>) {
        return compute_summary_sparse(storage);
    } else {
        SummaryStats ss;
        ss.num_haplotypes = storage.num_haplotypes();
        ss.num_sites      = storage.num_sites();
        ss.sfs            = compute_sfs(storage);
        ss.segregating_sites = segregating_sites_from_sfs(ss.sfs);
        ss.pi             = compute_pi_from_sfs(ss.sfs);
        ss.theta_w        = compute_theta_w(ss.segregating_sites, ss.num_haplotypes);
        ss.tajimas_d      = compute_tajimas_d(ss.pi, ss.theta_w,
                                               ss.segregating_sites, ss.num_haplotypes);
        ss.mean_heterozygosity = mean_heterozygosity(storage);
        return ss;
    }
}

}  // namespace gensim
