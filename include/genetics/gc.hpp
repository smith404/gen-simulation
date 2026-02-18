// =============================================================================
// gc.hpp — Garbage collection for fixed and lost mutations.
//
// Over many generations, the AlleleTable grows without bound because every
// mutation ever created is retained.  Most mutations are lost quickly by
// drift; some reach fixation (frequency 2N/2N).  Neither category needs
// active tracking:
//
//   - Lost (freq 0): the AlleleID is no longer present in any haplotype.
//     We can't shrink AlleleTable (IDs are indices), but we can mark them.
//
//   - Fixed (freq 2N): every haplotype carries this allele.  We record
//     it as a "substitution," reset those sites to kRefAllele, and fold
//     the fixed allele's fitness effect into a baseline offset.
//
// For DenseHaplotypes, we scan column-by-column.  For SparseVariants, we
// iterate the variant list and remove entries that are all-ref or all-same.
//
// Call gc_sweep() periodically (e.g., every 10–50 generations).
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <vector>

namespace gensim {

// ── Substitution record ─────────────────────────────────────────────────────
/// Represents a mutation that reached fixation and was removed from active
/// tracking.  Useful for downstream analysis.
struct Substitution {
    AlleleID   allele_id;
    SiteIndex  site;
    Generation fixation_gen;
};

// ── GC result ───────────────────────────────────────────────────────────────
struct GCResult {
    std::size_t                 lost_count   = 0;   // alleles at freq 0
    std::size_t                 fixed_count  = 0;   // alleles at freq 2N (fixations)
    double                      fitness_offset = 0.0; // accumulated fitness from fixed alleles
    std::vector<Substitution>   substitutions;        // newly fixed mutations
};

// =============================================================================
// Dense GC sweep
// =============================================================================

inline GCResult gc_sweep_dense(DenseHaplotypes& storage,
                                 const AlleleTable& table,
                                 Generation current_gen)
{
    GCResult result;
    const std::size_t H = storage.num_haplotypes();
    const std::size_t L = storage.num_sites();

    // For each site, check if a non-ref allele is fixed (present in all H
    // haplotypes with the same ID).
    for (std::size_t s = 0; s < L; ++s) {
        AlleleID first = storage.hap_ptr(0)[s];
        if (first == kRefAllele) continue;  // site is at least partly ref

        bool all_same = true;
        for (std::size_t h = 1; h < H; ++h) {
            if (storage.hap_ptr(h)[s] != first) {
                all_same = false;
                break;
            }
        }

        if (all_same) {
            // This allele is fixed.  Record the substitution and reset to ref.
            result.substitutions.push_back(
                Substitution{first, s, current_gen});
            result.fitness_offset += table[first].sel_coeff;
            ++result.fixed_count;

            // Reset all haplotypes at this site to reference.
            for (std::size_t h = 0; h < H; ++h) {
                storage.hap_ptr(h)[s] = kRefAllele;
            }
        }
    }

    return result;
}

// =============================================================================
// Sparse GC sweep
// =============================================================================

inline GCResult gc_sweep_sparse(SparseVariants& storage,
                                  const AlleleTable& table,
                                  Generation current_gen)
{
    GCResult result;
    auto& vars = storage.variants();
    const std::size_t H = storage.num_haplotypes();

    // Walk variants in reverse so we can erase without invalidating indices.
    for (auto it = vars.begin(); it != vars.end(); /* in body */) {
        // Check: is every haplotype the same allele?
        bool all_ref = true;
        bool any_ref = false;
        AlleleID first_non_ref = kRefAllele;
        bool all_same_non_ref = true;

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID a = it->alleles[h];
            if (a == kRefAllele) {
                any_ref = true;
            } else {
                all_ref = false;
                if (first_non_ref == kRefAllele) {
                    first_non_ref = a;
                } else if (a != first_non_ref) {
                    all_same_non_ref = false;
                }
            }
        }

        if (all_ref) {
            // Lost variant — all haplotypes reverted to reference.
            ++result.lost_count;
            it = vars.erase(it);
        } else if (!any_ref && all_same_non_ref) {
            // Fixed: every haplotype carries the same non-ref allele.
            result.substitutions.push_back(
                Substitution{first_non_ref, it->pos, current_gen});
            result.fitness_offset += table[first_non_ref].sel_coeff;
            ++result.fixed_count;
            it = vars.erase(it);
        } else {
            ++it;
        }
    }

    return result;
}

}  // namespace gensim
