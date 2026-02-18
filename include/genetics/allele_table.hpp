// =============================================================================
// allele_table.hpp — Global registry that maps AlleleID → metadata.
//
// Thread-safety note: the table is append-only during a simulation.  If you
// parallelise offspring production across threads, either
//   (a) give each thread a local AlleleTable and merge post-generation, or
//   (b) protect new_allele() with a mutex / atomic next_id_.
// The single-threaded path shown here is lock-free.
// =============================================================================
#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace gensim {

// ── Per-allele metadata ─────────────────────────────────────────────────────
struct AlleleInfo {
    Generation  origin_gen   = 0;       // generation the allele first appeared
    Fitness     sel_coeff    = 0.0;     // selection coefficient (0 = neutral)
    double      dominance_h  = 0.5;    // dominance coefficient (0=recessive, 0.5=codominant, 1=dominant)
    std::string label        = {};     // optional human-readable tag
};

// ── AlleleTable ─────────────────────────────────────────────────────────────
// Stores metadata for every allele ever created.  AlleleID 0 is always the
// reference allele and is inserted in the constructor.
class AlleleTable {
public:
    explicit AlleleTable(std::size_t reserve_hint = 4096) {
        entries_.reserve(reserve_hint);
        // ID 0 = reference / ancestral allele.
        entries_.push_back(AlleleInfo{0, 0.0, 0.5, "ref"});
    }

    // ── Create a new allele and return its unique ID ────────────────────────
    [[nodiscard]] AlleleID new_allele(AlleleInfo info) {
        AlleleID id = static_cast<AlleleID>(entries_.size());
        entries_.push_back(std::move(info));
        return id;
    }

    // ── Query ───────────────────────────────────────────────────────────────
    [[nodiscard]] const AlleleInfo& operator[](AlleleID id) const noexcept {
        return entries_[id];
    }

    [[nodiscard]] AlleleInfo& operator[](AlleleID id) noexcept {
        return entries_[id];
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return entries_.size();
    }

    // ── Thread-safe note ────────────────────────────────────────────────────
    // For parallel mutation, replace entries_ with a concurrent vector or
    // protect new_allele() with std::mutex.  The ID can also be generated
    // via std::atomic<AlleleID> next_id_.

private:
    std::vector<AlleleInfo> entries_;
};

}  // namespace gensim
