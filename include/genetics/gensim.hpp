// =============================================================================
// gensim.hpp — Single-include convenience header for the genetics simulator.
//
// Including this file brings in the entire library.  For finer control,
// include individual headers.
//
//   #include "genetics/gensim.hpp"   // everything
//
// Or pick what you need:
//
//   #include "genetics/types.hpp"
//   #include "genetics/dense_haplotypes.hpp"
//   #include "genetics/population.hpp"
//   #include "genetics/simulation.hpp"
//   ...
// =============================================================================
#pragma once

// ── Core types & concepts ───────────────────────────────────────────────────
#include "types.hpp"
#include "concepts.hpp"

// ── Allele registry ─────────────────────────────────────────────────────────
#include "allele_table.hpp"

// ── Storage backends ────────────────────────────────────────────────────────
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"

// ── Distribution of Fitness Effects ─────────────────────────────────────────
#include "dfe.hpp"

// ── Policy objects ──────────────────────────────────────────────────────────
#include "mutation_model.hpp"
#include "recombination_model.hpp"
#include "fitness_model.hpp"
#include "selection.hpp"
#include "mating_system.hpp"

// ── Multi-chromosome & quantitative genetics ────────────────────────────────
#include "chromosome.hpp"
#include "quantitative_trait.hpp"
#include "mutation_rate_map.hpp"

// ── Population container ────────────────────────────────────────────────────
#include "population.hpp"

// ── Demographics & migration ────────────────────────────────────────────────
#include "demographics.hpp"
#include "migration.hpp"

// ── Simulation drivers ──────────────────────────────────────────────────────
#include "events.hpp"
#include "simulation.hpp"

// ── Analysis ────────────────────────────────────────────────────────────────
#include "statistics.hpp"
#include "ld.hpp"
#include "fst.hpp"

// ── Housekeeping ────────────────────────────────────────────────────────────
#include "gc.hpp"
#include "ancestry.hpp"
#include "serialization.hpp"

// ── Output formats ──────────────────────────────────────────────────────────
#include "output_formats.hpp"
