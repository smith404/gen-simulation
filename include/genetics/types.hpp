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
#include <vector>

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

// ── Sex ─────────────────────────────────────────────────────────────────────
enum class Sex : std::uint8_t {
    Female = 0,
    Male   = 1,
    Hermaphrodite = 2   // for selfing / monoecious species
};

// ── Chromosome identifier ───────────────────────────────────────────────────
using ChromosomeID = std::uint32_t;

// ── Chromosome type ─────────────────────────────────────────────────────────
enum class ChromosomeType : std::uint8_t {
    Autosome    = 0,
    X           = 1,   // X (or Z in ZW systems)
    Y           = 2,   // Y (or W in ZW system)
    Mitochondrial = 3  // haploid, maternally inherited
};

// ── Spatial coordinate ──────────────────────────────────────────────────────
struct Position2D {
    double x = 0.0;
    double y = 0.0;
};

// ── Trait value (for quantitative genetics) ─────────────────────────────────
using TraitValue = double;

// ── Per-individual metadata ─────────────────────────────────────────────────
/// Lightweight per-individual record.  Carries an index into the haplotype
/// storage (haplotypes 2*i and 2*i+1) plus optional demographic and trait
/// metadata.  This struct lives in types.hpp so that mating systems,
/// quantitative trait models, and other headers can use it without
/// including population.hpp.
struct Individual {
    std::size_t index = 0;      // index into the storage (haplotypes 2*i, 2*i+1)
    Sex         sex   = Sex::Hermaphrodite;
    std::uint16_t age = 0;      // 0 = newborn; incremented each generation in nonWF models
    Position2D  position{};     // spatial location (unused if not spatial)
    DemeID      deme  = kNoDeme;

    // ── Quantitative trait values ───────────────────────────────────────────
    // Populated by a QuantitativeTraitPolicy each generation.  One entry per
    // trait dimension (e.g. height, fecundity, parasite resistance, …).
    std::vector<TraitValue> traits;

    // ── Custom floating-point info fields ────────────────────────────────────
    // Arbitrary per-individual metadata (energy, migration tag, etc.).
    std::vector<double> info_fields;
};

}  // namespace gensim
