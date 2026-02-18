// =============================================================================
// types.hpp — Core type aliases and constants for the genetics simulator.
//
// Centralises every fundamental type so that width changes (e.g. moving to
// 32-bit allele IDs) propagate automatically.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace gensim {

// ── Allele representation ───────────────────────────────────────────────────
// AlleleID is a lightweight integer tag.  0 is reserved for the reference
// (ancestral) allele; new mutations receive successive IDs via AlleleTable.
using AlleleID = std::uint32_t;

/// Sentinel value indicating "no allele" / uninitialised.
inline constexpr AlleleID kNoAllele = std::numeric_limits<AlleleID>::max();

/// The reference (ancestral) allele present at every site before mutation.
inline constexpr AlleleID kRefAllele = 0;

// ── Site / position index ───────────────────────────────────────────────────
using SiteIndex = std::size_t;

// ── Generation counter ──────────────────────────────────────────────────────
using Generation = std::uint64_t;

// ── Fitness value ───────────────────────────────────────────────────────────
using Fitness = double;

// ── Deme (subpopulation) identifier ─────────────────────────────────────────
using DemeID = std::uint32_t;

/// Sentinel for "no deme" / single-population mode.
inline constexpr DemeID kNoDeme = std::numeric_limits<DemeID>::max();

}  // namespace gensim
