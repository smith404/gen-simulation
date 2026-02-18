// =============================================================================
// dense_haplotypes.hpp — Contiguous-memory haplotype storage backend.
//
// Layout: a flat std::vector<AlleleID> of shape (2*N, L) stored in row-major
// order.  Each row is one haploid chromosome; individual i owns rows 2*i and
// 2*i+1.  This layout is cache-friendly for per-haplotype operations
// (recombination, mutation) and allows block copies via std::memcpy.
//
// Double-buffering: two identically-sized buffers are maintained.  Offspring
// are written into the "back" buffer while parents are read from the "front".
// At generation end, swap_buffers() makes the offspring the new parents.
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>     // std::memcpy
#include <stdexcept>
#include <vector>

namespace gensim {

class DenseHaplotypes {
public:
    // ── Construction ────────────────────────────────────────────────────────
    // N = number of diploid individuals, L = number of sites.
    DenseHaplotypes() = default;

    DenseHaplotypes(std::size_t num_individuals, std::size_t num_sites)
        : n_{num_individuals}, l_{num_sites}
    {
        const std::size_t total = 2 * n_ * l_;
        front_.assign(total, kRefAllele);   // all ancestral
        back_.assign(total, kRefAllele);
    }

    // ── HaplotypeStorage concept interface ──────────────────────────────────

    [[nodiscard]] std::size_t num_individuals() const noexcept { return n_; }
    [[nodiscard]] std::size_t num_sites()       const noexcept { return l_; }
    [[nodiscard]] std::size_t num_haplotypes()  const noexcept { return 2 * n_; }

    /// Pointer to the first site of haplotype `hap` in the FRONT (parent) buffer.
    [[nodiscard]] const AlleleID* hap_ptr(std::size_t hap) const noexcept {
        assert(hap < 2 * n_);
        return front_.data() + hap * l_;
    }

    [[nodiscard]] AlleleID* hap_ptr(std::size_t hap) noexcept {
        assert(hap < 2 * n_);
        return front_.data() + hap * l_;
    }

    /// Pointer into the BACK (offspring) buffer — used during offspring creation.
    [[nodiscard]] AlleleID* offspring_hap_ptr(std::size_t hap) noexcept {
        assert(hap < 2 * n_);
        return back_.data() + hap * l_;
    }

    [[nodiscard]] const AlleleID* offspring_hap_ptr(std::size_t hap) const noexcept {
        assert(hap < 2 * n_);
        return back_.data() + hap * l_;
    }

    // ── Block copy helpers (used by recombination) ──────────────────────────

    /// Copy a contiguous segment [site_begin, site_end) from a parent haplotype
    /// in the front buffer into an offspring haplotype in the back buffer.
    void copy_segment(std::size_t src_hap, std::size_t dst_hap,
                      std::size_t site_begin, std::size_t site_end) noexcept
    {
        assert(site_end <= l_ && site_begin <= site_end);
        const std::size_t len = site_end - site_begin;
        if (len == 0) return;
        std::memcpy(offspring_hap_ptr(dst_hap) + site_begin,
                    hap_ptr(src_hap) + site_begin,
                    len * sizeof(AlleleID));
    }

    // ── Buffer management ───────────────────────────────────────────────────

    /// Swap front/back so that offspring become the current generation.
    void swap_buffers() noexcept { front_.swap(back_); }

    // ── Resize (e.g. population-size change — not needed in basic sim) ──────

    void resize(std::size_t new_n, std::size_t new_l) {
        n_ = new_n;
        l_ = new_l;
        const std::size_t total = 2 * n_ * l_;
        front_.assign(total, kRefAllele);
        back_.assign(total, kRefAllele);
    }

    // ── Statistics helpers ──────────────────────────────────────────────────

    /// Count how many sites have at least one non-reference allele.
    [[nodiscard]] std::size_t count_segregating_sites() const noexcept {
        std::size_t seg = 0;
        const std::size_t H = 2 * n_;
        // PERF: tiling by small blocks of sites here improves cache behaviour
        // when L is very large (e.g. > 1M).  Left as a loop for clarity.
        for (std::size_t s = 0; s < l_; ++s) {
            bool has_alt = false;
            for (std::size_t h = 0; h < H; ++h) {
                if (front_[h * l_ + s] != kRefAllele) { has_alt = true; break; }
            }
            seg += has_alt;
        }
        return seg;
    }

    /// Return a histogram of allele counts at site `s`.
    [[nodiscard]] std::vector<std::pair<AlleleID, std::size_t>>
    allele_counts(std::size_t site) const {
        assert(site < l_);
        // Small map implemented as sorted vector — fine for low allele counts.
        std::vector<std::pair<AlleleID, std::size_t>> counts;
        const std::size_t H = 2 * n_;
        for (std::size_t h = 0; h < H; ++h) {
            AlleleID a = front_[h * l_ + site];
            auto it = std::find_if(counts.begin(), counts.end(),
                                   [a](auto& p){ return p.first == a; });
            if (it != counts.end()) ++it->second;
            else counts.emplace_back(a, 1);
        }
        return counts;
    }

private:
    std::size_t n_ = 0;   // diploid individuals
    std::size_t l_ = 0;   // sites
    std::vector<AlleleID> front_;
    std::vector<AlleleID> back_;
};

}  // namespace gensim
