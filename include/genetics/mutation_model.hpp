// =============================================================================
// mutation_model.hpp — Pluggable mutation policies.
//
// Two concrete models are provided:
//
//   1. BernoulliInfiniteAlleles  – per-site independent Bernoulli trial with
//      probability μ.  Each new mutation receives a globally unique AlleleID
//      from the AlleleTable (infinite-alleles model).
//
//   2. BernoulliFiniteAlleles   – same Bernoulli process, but alleles are drawn
//      uniformly from {0, 1, …, K-1} (finite-alleles / K-alleles model).
//
// Both models are stateless aside from the mutation rate and can be passed as
// lightweight objects.  They are templated on nothing — the Simulation driver
// dispatches to the correct overload depending on the storage backend.
//
// Extension point: users can write any callable with signature
//   void operator()(Storage&, AlleleTable&, Generation, RNG&) const;
// and pass it as a mutation policy.
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "dfe.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <cmath>
#include <random>

namespace gensim {

// ─────────────────────────────────────────────────────────────────────────────
// BernoulliInfiniteAlleles
// ─────────────────────────────────────────────────────────────────────────────
// Each site on each offspring haplotype independently mutates with probability
// `mu`.  Every mutation is a brand-new allele (infinite-alleles assumption).
//
// Performance note (Dense):  When μ is small, the expected number of mutations
// per haplotype is μ·L.  Rather than flipping L coins we draw the number of
// mutations from Binomial(L, μ) and scatter them, which is O(μ·L) per
// haplotype instead of O(L).  For very small μ·L we approximate with Poisson.
// ─────────────────────────────────────────────────────────────────────────────
struct BernoulliInfiniteAlleles {
    double mu = 1e-5;  // per-site, per-generation mutation probability

    // ── Dense backend ───────────────────────────────────────────────────────
    void operator()(DenseHaplotypes& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        // Use Poisson approximation when lambda is small; otherwise Binomial.
        // Both give the total number of mutations per haplotype.
        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<std::size_t> site_dist(0, L - 1);

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID* row = storage.offspring_hap_ptr(h);
            std::size_t n_mut = pois(rng);

            for (std::size_t m = 0; m < n_mut; ++m) {
                std::size_t site = site_dist(rng);
                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, 0.0, 0.5, {}});
                row[site] = new_id;
            }
        }
    }

    // ── Sparse backend ──────────────────────────────────────────────────────
    void operator()(SparseVariants& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<SiteIndex> site_dist(0, L - 1);

        for (std::size_t h = 0; h < H; ++h) {
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                SiteIndex site = site_dist(rng);
                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, 0.0, 0.5, {}});
                storage.offspring_set(h, site, new_id);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// BernoulliFiniteAlleles
// ─────────────────────────────────────────────────────────────────────────────
// Like BernoulliInfiniteAlleles, but the mutant allele is drawn uniformly
// from {0, …, K-1}.  Useful for modelling e.g. nucleotide states (K=4).
// ─────────────────────────────────────────────────────────────────────────────
struct BernoulliFiniteAlleles {
    double      mu = 1e-5;
    AlleleID    K  = 4;        // number of possible alleles (0..K-1)

    void operator()(DenseHaplotypes& storage, AlleleTable& /* unused */,
                    Generation /* gen */, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<std::size_t> site_dist(0, L - 1);
        std::uniform_int_distribution<AlleleID> allele_dist(0, K - 1);

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID* row = storage.offspring_hap_ptr(h);
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                std::size_t site = site_dist(rng);
                AlleleID new_a = allele_dist(rng);
                // Ensure mutation produces a *different* allele.
                while (new_a == row[site]) new_a = allele_dist(rng);
                row[site] = new_a;
            }
        }
    }

    void operator()(SparseVariants& storage, AlleleTable& /* unused */,
                    Generation /* gen */, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<SiteIndex> site_dist(0, L - 1);
        std::uniform_int_distribution<AlleleID> allele_dist(0, K - 1);

        for (std::size_t h = 0; h < H; ++h) {
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                SiteIndex site = site_dist(rng);
                AlleleID cur = storage.offspring_get(h, site);
                AlleleID new_a = allele_dist(rng);
                while (new_a == cur) new_a = allele_dist(rng);
                storage.offspring_set(h, site, new_a);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DFEMutation — infinite-alleles with a Distribution of Fitness Effects.
// ─────────────────────────────────────────────────────────────────────────────
// Each new mutation's selection coefficient is drawn from a DFE, and its
// dominance coefficient is set to `default_h`.  This formalises the ad-hoc
// "DeleteriousMutation" wrapper from the original demo.
// ─────────────────────────────────────────────────────────────────────────────
struct DFEMutation {
    double     mu        = 1e-5;      // per-site mutation probability
    DFEVariant dfe       = GammaDFE{-0.01, 0.3};  // DFE for selection coefficient
    double     default_h = 0.5;       // default dominance coefficient for new mutations

    // ── Dense backend ───────────────────────────────────────────────────────
    void operator()(DenseHaplotypes& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<std::size_t> site_dist(0, L - 1);

        // Local copy of DFE for drawing (std distributions mutate internal state).
        DFEVariant local_dfe = dfe;

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID* row = storage.offspring_hap_ptr(h);
            std::size_t n_mut = pois(rng);

            for (std::size_t m = 0; m < n_mut; ++m) {
                std::size_t site = site_dist(rng);
                double s = draw_s(local_dfe, rng);
                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, s, default_h, {}});
                row[site] = new_id;
            }
        }
    }

    // ── Sparse backend ──────────────────────────────────────────────────────
    void operator()(SparseVariants& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        const std::size_t L = storage.num_sites();
        const double lambda = mu * static_cast<double>(L);

        std::poisson_distribution<std::size_t> pois(lambda);
        std::uniform_int_distribution<SiteIndex> site_dist(0, L - 1);
        DFEVariant local_dfe = dfe;

        for (std::size_t h = 0; h < H; ++h) {
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                SiteIndex site = site_dist(rng);
                double s = draw_s(local_dfe, rng);
                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, s, default_h, {}});
                storage.offspring_set(h, site, new_id);
            }
        }
    }
};

}  // namespace gensim
