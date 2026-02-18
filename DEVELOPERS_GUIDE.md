# gensim Developer's Guide

A comprehensive reference for developers extending, integrating, or contributing to the **gensim** forward-in-time diploid population genetics simulation library.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Build & Toolchain](#2-build--toolchain)
3. [Core Type System](#3-core-type-system)
4. [C++20 Concepts — The Contract Layer](#4-c20-concepts--the-contract-layer)
5. [Allele Registry](#5-allele-registry)
6. [Storage Backends](#6-storage-backends)
7. [Distribution of Fitness Effects (DFE)](#7-distribution-of-fitness-effects-dfe)
8. [Mutation Policies](#8-mutation-policies)
9. [Recombination Policies](#9-recombination-policies)
10. [Fitness Policies](#10-fitness-policies)
11. [Selection Policies](#11-selection-policies)
12. [Mating System Policies](#12-mating-system-policies)
13. [Multi-Chromosome Genomes](#13-multi-chromosome-genomes)
14. [Quantitative Trait Models](#14-quantitative-trait-models)
15. [Variable Mutation Rate Maps](#15-variable-mutation-rate-maps)
16. [Population Container](#16-population-container)
17. [Simulation Drivers](#17-simulation-drivers)
18. [Event / Hook System](#18-event--hook-system)
19. [Demographics](#19-demographics)
20. [Migration & Demes](#20-migration--demes)
21. [Statistics](#21-statistics)
22. [Linkage Disequilibrium](#22-linkage-disequilibrium)
23. [Population Differentiation (Fst)](#23-population-differentiation-fst)
24. [Garbage Collection](#24-garbage-collection)
25. [Ancestry Tracking](#25-ancestry-tracking)
26. [Serialization / Checkpointing](#26-serialization--checkpointing)
27. [Output Formats](#27-output-formats)
28. [Writing Custom Policies](#28-writing-custom-policies)
29. [Performance Considerations](#29-performance-considerations)
30. [Thread Safety](#30-thread-safety)
31. [Extending the Library](#31-extending-the-library)
32. [Troubleshooting & Common Pitfalls](#32-troubleshooting--common-pitfalls)

---

## 1. Architecture Overview

gensim is a **header-only C++20 library** that uses **compile-time polymorphism** throughout — no virtual functions in the hot path.  Every configurable aspect (storage, mutation, recombination, fitness, selection, mating) is a template parameter or lightweight policy object.

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│  (configure policies, build Population, run Simulation)          │
└──────────────┬───────────────────────────────────────────────────┘
               │
       ┌───────▼────────┐     ┌─────────────────┐
       │  Simulation /  │────▶│ SimulationEvents │ (pre/post hooks)
       │  Advanced      │     └─────────────────┘
       │  Simulation    │────▶ DemographicSchedule
       │                │────▶ AncestryTracker
       └───────┬────────┘────▶ MatingSystem
               │
       ┌───────▼────────┐
       │   Population   │     Policy objects:
       │   <Storage,    │       • MutationPolicy
       │    Mut, Rec,   │       • RecombinationPolicy
       │    Fit>        │       • FitnessPolicy
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │    Storage     │     DenseHaplotypes
       │    Backend     │     SparseVariants
       │                │     (or custom)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  AlleleTable   │     Shared, append-only allele registry
       └────────────────┘
```

### Key design principles

| Principle | How it manifests |
|-----------|-----------------|
| **Zero-cost abstraction** | Policy objects are structs with `operator()` — fully inlined. No vtable. |
| **Double-buffered storage** | Parents (front buffer, read-only) ↔ Offspring (back buffer, write-only). `swap_buffers()` is O(1). |
| **Concept-gated generics** | Every pluggable interface has a C++20 concept. Custom types that satisfy the concept are automatically usable. |
| **Composability** | Mix and match any combination of storage × mutation × recombination × fitness × selection × mating policies. |
| **Shared allele table** | `AlleleTable` is passed by reference, enabling multi-population scenarios (migration, metapopulations). |

### Header dependency graph (simplified)

```
types.hpp ──────┐
                ▼
concepts.hpp ───┤
                ▼
allele_table.hpp ────────┐
                         ▼
dense_haplotypes.hpp ────┤◀── sparse_variants.hpp
                         ▼
dfe.hpp ─────────────────┤
                         ▼
mutation_model.hpp ──────┤
recombination_model.hpp ─┤
fitness_model.hpp ───────┤
selection.hpp ───────────┤
mating_system.hpp ───────┤
                         ▼
population.hpp ──────────┤
                         ▼
events.hpp ──────────────┤
demographics.hpp ────────┤
ancestry.hpp ────────────┤
gc.hpp ──────────────────┤
                         ▼
simulation.hpp ──────────┤
                         ▼
statistics.hpp ──────────┤
ld.hpp ──────────────────┤
fst.hpp ─────────────────┤
serialization.hpp ───────┤
chromosome.hpp ──────────┤
quantitative_trait.hpp ──┤
mutation_rate_map.hpp ───┤
output_formats.hpp ──────┘
```

All headers are included by **`gensim.hpp`** for single-include convenience.

---

## 2. Build & Toolchain

### Requirements

- **Language standard**: C++20
- **No external dependencies**
- **Header-only** — nothing to link

### Compiler support

| Compiler | Minimum version | Command |
|----------|----------------|---------|
| MSVC | 2022 (v19.30+) | `cl /O2 /std:c++20 /EHsc /Iinclude src\main.cpp /Fe:gensim.exe` |
| GCC | 12+ | `g++ -O3 -std=c++20 -Iinclude -o gensim src/main.cpp` |
| Clang | 15+ | `clang++ -O3 -std=c++20 -Iinclude -o gensim src/main.cpp` |

### VS Code IntelliSense

If the C/C++ extension shows concept errors, configure `.vscode/c_cpp_properties.json`:

```json
{
    "configurations": [{
        "name": "Win32",
        "includePath": ["${workspaceFolder}/include/**"],
        "cStandard": "c17",
        "cppStandard": "c++20",
        "compilerPath": "cl.exe",
        "intelliSenseMode": "msvc-x86"
    }],
    "version": 4
}
```

---

## 3. Core Type System

**Header:** `types.hpp`

All fundamental types are centralised here so that width changes propagate automatically.

### Type aliases

| Alias | Underlying type | Purpose |
|-------|----------------|---------|
| `AlleleID` | `uint32_t` | Lightweight integer tag for alleles |
| `SiteIndex` | `size_t` | Position along the genome |
| `Generation` | `uint64_t` | Generation counter |
| `Fitness` | `double` | Individual fitness value |
| `DemeID` | `uint32_t` | Subpopulation identifier |
| `ChromosomeID` | `uint32_t` | Chromosome identifier |
| `TraitValue` | `double` | Quantitative trait phenotype value |

### Sentinel constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `kNoAllele` | `AlleleID::max()` | Uninitialised / missing allele |
| `kRefAllele` | `0` | Ancestral / reference allele |
| `kNoDeme` | `DemeID::max()` | Single-population mode (no deme assignment) |

### Enumerations

```cpp
enum class Sex : uint8_t { Female = 0, Male = 1, Hermaphrodite = 2 };
enum class ChromosomeType : uint8_t { Autosome = 0, X = 1, Y = 2, Mitochondrial = 3 };
```

### Structs

**`Position2D`** — 2D spatial coordinate for landscape simulations:
```cpp
struct Position2D { double x = 0.0; double y = 0.0; };
```

**`Individual`** — per-individual metadata record:
```cpp
struct Individual {
    size_t      index    = 0;               // haplotypes at 2*index, 2*index+1
    Sex         sex      = Sex::Hermaphrodite;
    uint16_t    age      = 0;               // 0 = newborn
    Position2D  position {};
    DemeID      deme     = kNoDeme;
    std::vector<TraitValue> traits;         // filled by trait models
    std::vector<double>     info_fields;    // arbitrary per-individual data
};
```

`Individual` lives in `types.hpp` (rather than `population.hpp`) so that mating systems, trait models, and output formatters can use it without circular includes.

---

## 4. C++20 Concepts — The Contract Layer

**Header:** `concepts.hpp` (core), `mating_system.hpp` (MatingSystemConcept)

Every pluggable interface is formalised as a C++20 concept, providing:
- **Documentation** — the required API surface is machine-readable
- **Diagnostics** — template errors become short, actionable messages
- **Extensibility** — third-party types satisfying a concept work automatically

### Concept reference

#### `HaplotypeStorage<S>`

The minimal interface every storage backend must satisfy.

```cpp
template <typename S>
concept HaplotypeStorage = requires(S s, const S cs, size_t idx, SiteIndex site) {
    { cs.num_individuals() } -> std::convertible_to<size_t>;
    { cs.num_sites()       } -> std::convertible_to<size_t>;
    { cs.num_haplotypes()  } -> std::convertible_to<size_t>;
    { s.copy_segment(idx, idx, site, site) };              // front → back block copy
    { s.swap_buffers() };                                   // O(1) swap
    { cs.count_segregating_sites() } -> std::convertible_to<size_t>;
};
```

#### `PointerAccessStorage<S>`

Extends `HaplotypeStorage` with contiguous per-haplotype memory access. Enables `memcpy`-based block copies and direct array iteration.

```cpp
template <typename S>
concept PointerAccessStorage = HaplotypeStorage<S> &&
    requires(S s, const S cs, size_t hap) {
        { cs.hap_ptr(hap)          } -> std::same_as<const AlleleID*>;
        { s.hap_ptr(hap)           } -> std::same_as<AlleleID*>;
        { s.offspring_hap_ptr(hap) } -> std::same_as<AlleleID*>;
    };
```

#### `ElementAccessStorage<S>`

Extends `HaplotypeStorage` with per-element get/set. Both built-in backends satisfy this.

```cpp
template <typename S>
concept ElementAccessStorage = HaplotypeStorage<S> &&
    requires(S s, const S cs, size_t hap, SiteIndex site, AlleleID a) {
        { cs.get(hap, site)             } -> std::convertible_to<AlleleID>;
        { cs.offspring_get(hap, site)   } -> std::convertible_to<AlleleID>;
        { s.offspring_set(hap, site, a) };
    };
```

#### Policy concepts

| Concept | Signature pattern |
|---------|-------------------|
| `MutationPolicyFor<M, S>` | `m(storage&, table&, gen, rng)` |
| `RecombinationPolicyFor<R, S>` | `r(parents, offspring, h0, h1, offspring_hap, rng)` |
| `FitnessPolicyFor<F, S>` | `f(storage, ind_idx, table) → Fitness` |
| `SelectionPolicyConcept<SP>` | `sp(fitnesses, rng) → size_t` |
| `MatingSystemConcept<MS>` | `ms(individuals, fitnesses, rng) → ParentPair` |
| `DFEConcept<D>` | `d(rng) → double` |

---

## 5. Allele Registry

**Header:** `allele_table.hpp`

The `AlleleTable` is the genome-wide registry that maps `AlleleID` → metadata. It is **shared by reference** across populations, enabling migration scenarios.

### AlleleInfo

```cpp
struct AlleleInfo {
    Generation origin_gen = 0;     // generation the mutation arose
    Fitness    sel_coeff  = 0.0;   // selection coefficient s
    double     dominance_h = 0.5;  // dominance coefficient h
    std::string label;             // optional human-readable label
};
```

### AlleleTable API

```cpp
class AlleleTable {
public:
    AlleleTable(size_t reserve_hint = 4096);

    AlleleID new_allele(AlleleInfo info);     // append-only, returns new ID
    AlleleInfo&       operator[](AlleleID);   // O(1) lookup (mutable)
    const AlleleInfo& operator[](AlleleID) const;
    size_t size() const noexcept;
};
```

**Important**: `AlleleID 0` is pre-inserted as the reference allele with `sel_coeff = 0`. All new alleles get successive IDs starting from 1.

**Thread-safety**: Not thread-safe by default. For parallel mutation, either use a mutex or give each thread a local table and merge afterward.

---

## 6. Storage Backends

### DenseHaplotypes

**Header:** `dense_haplotypes.hpp`

A flat `std::vector<AlleleID>` of shape `(2N, L)` in row-major order. Every cell stores an allele (even reference alleles). Best when L is moderate or the mutation rate is high.

```cpp
class DenseHaplotypes {
public:
    DenseHaplotypes(size_t num_individuals, size_t num_sites);

    // Size queries
    size_t num_individuals() const;     // N
    size_t num_sites() const;           // L
    size_t num_haplotypes() const;      // 2N

    // Element access (front buffer = parents)
    AlleleID  get(size_t hap, SiteIndex pos) const;
    AlleleID* hap_ptr(size_t hap);                        // raw row pointer
    const AlleleID* hap_ptr(size_t hap) const;

    // Element access (back buffer = offspring)
    AlleleID  offspring_get(size_t hap, SiteIndex pos) const;
    void      offspring_set(size_t hap, SiteIndex pos, AlleleID a);
    AlleleID* offspring_hap_ptr(size_t hap);

    // Block copy (front → back)
    void copy_segment(size_t src_hap, size_t dst_hap,
                      SiteIndex begin, SiteIndex end);    // uses memcpy

    // Lifecycle
    void swap_buffers();                                  // O(1) vector::swap
    void resize(size_t new_n, size_t new_l);

    // Statistics
    size_t count_segregating_sites() const;
    std::vector<std::pair<AlleleID, size_t>> allele_counts(SiteIndex site) const;
};
```

**Satisfies:** `PointerAccessStorage` ∧ `ElementAccessStorage`

**Memory:** `2 × 2N × L × sizeof(AlleleID)` = `16NL` bytes (two buffers).

### SparseVariants

**Header:** `sparse_variants.hpp`

Stores only non-reference sites. Each `Variant` holds a position and a vector of alleles for all haplotypes at that site.

```cpp
struct Variant {
    SiteIndex pos;
    std::vector<AlleleID> alleles;  // length 2N, one per haplotype
};

class SparseVariants {
public:
    SparseVariants(size_t num_individuals, size_t num_sites);

    // Element access: O(log V) per call (binary search)
    AlleleID get(size_t hap, SiteIndex pos) const;
    void     set(size_t hap, SiteIndex pos, AlleleID a);
    AlleleID offspring_get(size_t hap, SiteIndex pos) const;
    void     offspring_set(size_t hap, SiteIndex pos, AlleleID a);

    void copy_segment(size_t src, size_t dst, SiteIndex begin, SiteIndex end);
    void clear_offspring_buffer();
    void swap_buffers();        // swap + clear back

    const std::vector<Variant>& variants() const;   // front buffer
    size_t count_segregating_sites() const;
};
```

**Satisfies:** `ElementAccessStorage` (but **not** `PointerAccessStorage` — no contiguous per-haplotype memory)

**Memory:** Proportional to number of segregating sites, not L.

### Choosing a backend

| Scenario | Recommended | Why |
|----------|-------------|-----|
| High μ, large L | `DenseHaplotypes` | amortised O(1) access, memcpy recombination |
| Low μ, large L, few segregating sites | `SparseVariants` | memory ∝ V instead of L |
| Need `hap_ptr()` for external tools | `DenseHaplotypes` | only Dense has contiguous rows |
| Custom | Implement `HaplotypeStorage` | any type satisfying the concept works |

---

## 7. Distribution of Fitness Effects (DFE)

**Header:** `dfe.hpp`

DFEs generate the selection coefficient `s` for each new mutation. They are used by `DFEMutation`, `RegionalDFEMutation`, and anywhere a random `s` is needed.

### Concrete DFE types

| Type | Constructor | Distribution |
|------|-------------|--------------|
| `FixedDFE` | `FixedDFE{s}` | Point mass at `s` |
| `ExponentialDFE` | `ExponentialDFE{mean_s}` | Exponential with sign from `mean_s` |
| `GammaDFE` | `GammaDFE{mean_s, shape}` | Gamma(shape, \|mean_s\|/shape); `shape < 1` = leptokurtic |
| `NormalDFE` | `NormalDFE{mean_s, stddev}` | Normal(mean_s, stddev²) |
| `UniformDFE` | `UniformDFE{lo, hi}` | Uniform on [lo, hi] |
| `MixtureDFE` | `MixtureDFE{components}` | Weighted mixture of any DFE types |

All satisfy `DFEConcept`: `d(rng) → double`.

### Type-erased variant

```cpp
using DFEVariant = std::variant<FixedDFE, ExponentialDFE, GammaDFE, NormalDFE, UniformDFE>;

// Draw from variant:
double s = draw_s(dfe_variant, rng);  // dispatched via std::visit
```

### Example: mixture DFE

```cpp
// 70% neutral, 20% slightly deleterious, 10% strongly deleterious
MixtureDFE dfe{{
    {0.7, FixedDFE{0.0}},
    {0.2, NormalDFE{-0.001, 0.0005}},
    {0.1, GammaDFE{-0.05, 0.3}}
}};
```

---

## 8. Mutation Policies

**Headers:** `mutation_model.hpp`, `mutation_rate_map.hpp`

Mutation policies apply new mutations to the offspring back buffer after recombination. All provide overloads for both `DenseHaplotypes` and `SparseVariants`.

### Standard mutation policies (`mutation_model.hpp`)

| Policy | Fields | Description |
|--------|--------|-------------|
| `BernoulliInfiniteAlleles` | `mu = 1e-5` | Each mutation creates a new unique `AlleleID`. Poisson-scattered. |
| `BernoulliFiniteAlleles` | `mu = 1e-5`, `K = 4` | Draws allele from {0..K−1}, ensuring change. |
| `DFEMutation` | `mu`, `dfe` (DFEVariant), `default_h = 0.5` | Infinite-alleles + DFE-drawn `sel_coeff` per mutation. |

**Signature:** `operator()(Storage&, AlleleTable&, Generation, mt19937_64&) const`

### Advanced mutation policies (`mutation_rate_map.hpp`)

| Policy | Fields | Description |
|--------|--------|-------------|
| `MappedMutation` | `rate_map` (MutationRateMap) | Uniform DFE with variable per-site rates. Sites chosen by cumulative weight binary search. |
| `RegionalDFEMutation` | `rate_map`, `type_map` (MutationTypeMap) | Different DFEs for different genomic regions (exons, introns, regulatory). |

### Performance: Poisson-scattered mutations

All policies use the Poisson approximation: instead of testing each of the L sites with probability μ, draw `k ~ Poisson(μ·L)` mutations per haplotype and place them at random sites. This makes the mutation step **O(μ·L·2N)** instead of **O(L·2N)** — a huge speedup when μ·L ≪ L.

---

## 9. Recombination Policies

**Header:** `recombination_model.hpp`

Recombination policies produce one offspring haplotype by copying alternating segments from two parental haplotypes.

### RecombinationMap

```cpp
struct RecombinationMap {
    std::vector<SiteIndex> endpoints;   // segment boundaries, sorted
    std::vector<double>    rates;       // per-segment recombination rate

    static RecombinationMap uniform(size_t L, double rate);
    double total_map_length() const;
    std::vector<double> cumulative_weights() const;
};
```

### Policies

| Policy | Fields | Description |
|--------|--------|-------------|
| `SingleCrossover` | — | One uniform random breakpoint in [1, L−1]. Uses `memcpy` for Dense. |
| `MultiBreakpoint` | `expected_breakpoints = 1.0` | Poisson-distributed number of crossovers at uniform positions. |
| `MappedCrossover` | `rmap` (RecombinationMap) | Variable rates along the genome — models hotspots and coldspots. |

**Signature:** `operator()(const Storage&, Storage&, hap0, hap1, offspring_hap, mt19937_64&) const`

### Example: recombination hotspot

```cpp
RecombinationMap rmap;
rmap.endpoints = {0, 800, 1200, 2000};
rmap.rates     = {1e-4, 1e-3, 1e-4};  // 10× hotspot at positions 800–1200
MappedCrossover rec{rmap};
```

---

## 10. Fitness Policies

**Header:** `fitness_model.hpp`

Fitness policies evaluate the fitness of a single individual from their genotype. They are called once per individual per generation by `Population::compute_fitness()`.

| Policy | Fields | Formula |
|--------|--------|---------|
| `NeutralFitness` | — | `w = 1.0` always |
| `AdditiveFitness` | — | `w = 1 + Σ contributions` (dominance-aware: het → h·s, hom → s) |
| `MultiplicativeFitness` | — | `w = Π(1 + contribution)` (dominance-aware) |
| `StabilizingSelectionFitness` | `optimum = 0.0`, `Vs = 1.0` | `w = exp(−(z − θ)² / (2·Vs))` |

**Signature:** `operator()(const Storage&, size_t individual, const AlleleTable&) const → Fitness`

**Dominance handling** (Additive & Multiplicative):
- Both haplotypes reference → contribution = 0
- Heterozygous (one allele non-ref) → contribution = h·s
- Homozygous derived (both same non-ref) → contribution = s
- Fitness is clamped to [0, ∞)

---

## 11. Selection Policies

**Header:** `selection.hpp`

Selection policies choose a parent index given a vector of fitnesses. They are used by the basic `Simulation` driver (directly) and as a fallback when no mating system is specified.

| Policy | Fields | Mechanism |
|--------|--------|-----------|
| `FitnessProportionate` | — | P(i) ∝ w(i). Classic roulette-wheel / Wright–Fisher. |
| `TournamentSelection` | `tournament_size = 3` | Pick k random individuals, fittest wins. k=1 ≡ random. |
| `TruncationSelection` | `fraction = 0.5` | Only top fraction reproduce; uniform among eligible. |
| `RankSelection` | `selection_pressure = 1.5` | P(rank r) ∝ 2−sp+2(sp−1)(r−1)/(N−1). sp ∈ [1,2]. |
| `BoltzmannSelection` | `temperature = 1.0` | P(i) ∝ exp(w(i)/T). Uses log-sum-exp for stability. |

**Signature:** `operator()(const std::vector<Fitness>&, mt19937_64&) const → size_t`

---

## 12. Mating System Policies

**Header:** `mating_system.hpp`

Mating system policies control how **pairs** of parents are formed. They sit between selection (which scores individuals) and recombination (which produces offspring genomes). The mating system is a template parameter of `AdvancedSimulation`.

### ParentPair

```cpp
struct ParentPair {
    size_t parent0;   // contributes haplotype 0 to offspring (maternal)
    size_t parent1;   // contributes haplotype 1 to offspring (paternal)
};
```

When `parent0 == parent1`, the offspring is selfed.

### Policies

| Policy | Fields | Description |
|--------|--------|-------------|
| `RandomMating` | — | Both parents drawn fitness-weighted independently. Default for Wright–Fisher. |
| `SexualMating` | — | Requires Male + Female. Parent0 = female, Parent1 = male. Maintains separate per-sex fitness distributions. |
| `SelfingMating` | `selfing_rate = 0.0` | With probability σ, offspring from one parent (p0 == p1). Otherwise two random parents. |
| `MonogamousMating` | — (stateful) | Each individual mates with at most one partner. Tracks paired set. Call `reset(N)` each generation. |
| `AssortativeMating` | `trait_index = 0`, `strength = 1.0` | Second parent weighted by exp(−strength·\|z_i − z_first\|). Requires `Individual::traits` to be filled. |
| `ClonalMating` | — | Asexual reproduction: p0 == p1 always. |
| `SexualSelfingMating` | `selfing_rate = 0.0`, `hermaphroditic = true` | Combined sexual + selfing pathway for monoecious species. |

**Signature:** `operator()(const vector<Individual>&, const vector<Fitness>&, mt19937_64&) const → ParentPair`

### Utility functions

```cpp
void assign_sex(std::vector<Individual>& inds, std::mt19937_64& rng,
                double proportion_male = 0.5);
void assign_hermaphrodite(std::vector<Individual>& inds);
```

### MatingSystemConcept

```cpp
template <typename MS>
concept MatingSystemConcept = requires(const MS ms,
                                        const std::vector<Individual>& inds,
                                        const std::vector<Fitness>& fit,
                                        std::mt19937_64& rng) {
    { ms(inds, fit, rng) } -> std::convertible_to<ParentPair>;
};
```

### Example: sexual mating simulation

```cpp
using Pop = Population<DenseHaplotypes, DFEMutation, MultiBreakpoint, AdditiveFitness>;
Pop pop(500, 5000, mut, rec, fit, table);
pop.assign_sex(0.5, rng);  // 50% male

AdvancedSimulation<Pop, FitnessProportionate, SexualMating> sim(pop, 42);
sim.run(100);
```

---

## 13. Multi-Chromosome Genomes

**Header:** `chromosome.hpp`

Real organisms have multiple chromosomes with free recombination between them. This module models multi-chromosome genome architecture with sex chromosome inheritance.

### ChromosomeSpec

```cpp
struct ChromosomeSpec {
    ChromosomeID    id        = 0;
    std::string     name;                 // e.g. "chr1", "chrX"
    size_t          num_sites = 0;
    ChromosomeType  type      = ChromosomeType::Autosome;
    double          recomb_rate = 1e-4;   // per-site, uniform within chromosome
    RecombinationMap recomb_map;          // optional detailed map

    RecombinationMap effective_map() const;  // returns explicit map or builds uniform
};
```

### GenomeSpec

```cpp
struct GenomeSpec {
    std::vector<ChromosomeSpec> chromosomes;

    size_t total_sites() const;
    size_t num_chromosomes() const;
    size_t chromosome_offset(size_t ci) const;  // flat offset for chromosome ci

    // Builders
    static GenomeSpec uniform_autosomes(size_t n, size_t sites_per_chrom,
                                         double recomb_rate = 1e-4);
    GenomeSpec& add_sex_chromosomes(size_t x_sites, size_t y_sites,
                                     double x_recomb = 1e-4, double y_recomb = 0.0);
    GenomeSpec& add_mitochondrial(size_t sites);
};
```

### MultiChromGenome

Wraps a single flat `DenseHaplotypes` with per-chromosome views. Each chromosome occupies a contiguous range of sites in the flat storage.

```cpp
class MultiChromGenome {
public:
    MultiChromGenome(size_t N, const GenomeSpec& spec);

    // Per-chromosome element access
    AlleleID get(size_t haplotype, size_t chrom_idx, SiteIndex site_in_chrom) const;
    void offspring_set(size_t haplotype, size_t chrom_idx, SiteIndex site, AlleleID a);

    // Per-chromosome recombination
    void recombine_chromosome(size_t ci, size_t parent_hap0, size_t parent_hap1,
                               size_t offspring_hap, mt19937_64& rng);

    // Full meiosis with independent assortment + sex chromosome rules
    void meiosis(size_t parent_idx, size_t offspring_hap,
                 Sex parent_sex, bool is_maternal, mt19937_64& rng);

    void swap_buffers();
    void prepare_offspring();
    DenseHaplotypes& storage();
    const GenomeSpec& spec() const;
};
```

### Sex chromosome inheritance rules

| Chromosome | Female → offspring | Male → offspring |
|------------|-------------------|-----------------|
| Autosome | Random orientation + recombination | Random orientation + recombination |
| X | Recombine both copies | Pass single X copy directly |
| Y | Fill with reference | Pass Y directly |
| Mitochondrial | Copy maternally | Fill with reference |

### Example: human-like genome

```cpp
auto genome = GenomeSpec::uniform_autosomes(22, 5000, 1e-4);
genome.add_sex_chromosomes(3000, 600);
genome.add_mitochondrial(500);

MultiChromGenome mcg(100, genome);  // 100 individuals
// During offspring production:
mcg.meiosis(parent_idx, offspring_hap_idx, parent_sex, is_maternal, rng);
```

---

## 14. Quantitative Trait Models

**Header:** `quantitative_trait.hpp`

Maps genotype → phenotype → fitness using the standard model: **P = G + E** where G is the genetic value and E is environmental noise drawn from N(0, σ²_E).

### Trait models

#### AdditiveTraitModel

```cpp
struct AdditiveTraitModel {
    double env_variance = 0.0;  // σ²_E

    // G = Σ sel_coeff across all non-ref alleles on both haplotypes
    // P = G + N(0, env_variance)
};
```

#### DominanceTraitModel

```cpp
struct DominanceTraitModel {
    double env_variance = 0.0;

    // Per-site contribution: hom ref → 0, het → h·s, hom derived → s
};
```

#### MultiTraitAdditiveModel

```cpp
struct MultiTraitAdditiveModel {
    size_t num_traits = 1;
    std::vector<double> env_variances;                  // one per trait
    std::vector<std::vector<double>> effects;           // effects[allele_id][trait]

    void set_effects(AlleleID a, std::vector<double> trait_effects);
    double get_effect(AlleleID a, size_t trait_idx) const;
    std::vector<double> genetic_values(const DenseHaplotypes&, size_t ind) const;
    std::vector<double> phenotypes(const DenseHaplotypes&, size_t ind, mt19937_64&) const;
};
```

### Trait-based fitness functions

| Function | Parameters | Formula |
|----------|-----------|---------|
| `StabilizingTraitFitness` | `trait_index=0`, `optimum=0`, `Vs=1` | w = exp(−(z−θ)²/(2Vs)) |
| `DirectionalTraitFitness` | `trait_index=0`, `beta=0.01` | w = max(1+βz, 0) |
| `CorrelationalTraitFitness` | `trait_1, trait_2, optimum_1, optimum_2, Vs, rho` | Two-trait Gaussian with correlation |

### Time-varying selection

```cpp
struct MovingOptimumSchedule {
    double initial_optimum = 0.0;
    double rate = 0.0;              // linear shift per generation
    double amplitude = 0.0;         // sinusoidal amplitude
    double period = 100.0;          // sinusoidal period
    double optimum_at(Generation g) const;
};
```

### Frequency-dependent fitness

```cpp
struct FrequencyDependentFitness {
    double strength = 0.1;
    size_t trait_index = 0;
    double bin_width = 0.1;
    // w_i = base * (1 + strength * (1 - freq_in_bin_i))
};
```

### TraitStatistics

```cpp
struct TraitStatistics {
    double mean, variance;       // mean phenotype, V_P
    double genetic_var;          // additive genetic variance V_A
    double env_var;              // environmental variance V_E
    double heritability;         // h² = V_A / V_P
    double min_val, max_val;
};

TraitStatistics compute_trait_statistics(const std::vector<Individual>& inds,
                                          size_t trait_index,
                                          double env_variance = 0.0);
```

### Integration with Population

```cpp
pop.compute_traits(my_trait_model, rng);
// Now every individual's traits vector is populated
auto stats = compute_trait_statistics(pop.individuals(), 0, 0.1);
```

---

## 15. Variable Mutation Rate Maps

**Header:** `mutation_rate_map.hpp`

### MutationRateMap

Piecewise-constant per-site mutation rates, analogous to `RecombinationMap`.

```cpp
struct MutationRateMap {
    std::vector<SiteIndex> endpoints;
    std::vector<double> rates;

    static MutationRateMap uniform(size_t L, double mu);
    double total_rate() const;
    double rate_at(SiteIndex site) const;
    std::vector<double> cumulative_weights() const;
};
```

### MutationTypeMap

Map genomic regions to mutation types and region-specific DFEs.

```cpp
enum class MutationTypeTag : uint8_t {
    Neutral = 0, Coding = 1, Regulatory = 2, Intergenic = 3, Custom = 4
};

struct MutationTypeRegion {
    SiteIndex start, end;
    MutationTypeTag tag;
    DFEVariant dfe;
    double dominance_h = 0.5;
};

struct MutationTypeMap {
    std::vector<MutationTypeRegion> regions;
    void add_region(SiteIndex start, SiteIndex end, MutationTypeTag tag,
                    DFEVariant dfe, double h = 0.5);
    const MutationTypeRegion* region_at(SiteIndex site) const;
};
```

### Example: exon/intron structure

```cpp
MutationRateMap rate_map;
rate_map.endpoints = {0, 500, 1000, 2000};
rate_map.rates     = {1e-5, 5e-5, 1e-5};  // higher rate in middle region

MutationTypeMap type_map;
type_map.add_region(0, 500, MutationTypeTag::Intergenic,
                    FixedDFE{0.0});                          // neutral
type_map.add_region(500, 1000, MutationTypeTag::Coding,
                    GammaDFE{-0.01, 0.3}, 0.3);             // deleterious, partially recessive
type_map.add_region(1000, 2000, MutationTypeTag::Regulatory,
                    NormalDFE{0.0, 0.001}, 0.5);            // weak effects

RegionalDFEMutation mut{rate_map, type_map};
```

---

## 16. Population Container

**Header:** `population.hpp`

`Population` is the central class that owns the storage, policies, and per-individual metadata.

```cpp
template <typename Storage,
          typename MutationPolicy,
          typename RecombPolicy,
          typename FitnessPolicy>
class Population {
public:
    Population(size_t N, size_t L,
               MutationPolicy mut, RecombPolicy rec, FitnessPolicy fit,
               AlleleTable& table);

    // Accessors
    size_t size() const;
    Storage&       storage();
    const Storage& storage() const;
    AlleleTable&       alleles();
    const AlleleTable& alleles() const;
    const std::vector<Fitness>& fitness() const;
    std::vector<Individual>&       individuals();
    const std::vector<Individual>& individuals() const;
    const MutationPolicy& mutation_policy() const;
    const RecombPolicy&   recomb_policy() const;
    const FitnessPolicy&  fitness_policy() const;

    // Lifecycle (called by Simulation)
    void compute_fitness();                          // fills fitness_cache_
    size_t select_parent(mt19937_64& rng) const;     // fitness-proportionate
    void recombine(size_t parent_ind, size_t offspring_hap, mt19937_64& rng);
    void mutate(Generation gen, mt19937_64& rng);
    void advance();                                  // swap_buffers()
    void prepare_offspring_buffer();                 // clears back buffer (sparse only)

    // Statistics
    double mean_fitness() const;
    size_t segregating_sites() const;

    // Quantitative traits
    template <typename TraitModel>
    void compute_traits(const TraitModel& model, mt19937_64& rng);

    // Sex assignment
    void assign_sex(double proportion_male, mt19937_64& rng);
};
```

### Typical usage

```cpp
AlleleTable table;
BernoulliInfiniteAlleles mut{1e-5};
SingleCrossover          rec;
NeutralFitness           fit;

using Pop = Population<DenseHaplotypes, decltype(mut), decltype(rec), decltype(fit)>;
Pop pop(1000, 10000, mut, rec, fit, table);
```

---

## 17. Simulation Drivers

**Header:** `simulation.hpp`

### Basic driver: `Simulation<PopType>`

Lightweight WF loop with a per-generation callback.

```cpp
template <typename PopType>
class Simulation {
public:
    Simulation(PopType& population, uint64_t seed);
    void run(Generation num_generations,
             GenerationCallback<PopType> callback = nullptr);
    void step(Generation g);
    Generation generation() const;
};
```

**Generation loop:**
1. `compute_fitness()`
2. For each offspring: select 2 parents via `discrete_distribution`, `recombine()` each
3. `mutate()`
4. `advance()` (swap buffers)

### Advanced driver: `AdvancedSimulation<PopType, SelectionPolicy, MatingSystem>`

Full-featured driver integrating all subsystems.

```cpp
template <typename PopType,
          typename SelectionPolicy = FitnessProportionate,
          typename MatingSystem    = RandomMating>
class AdvancedSimulation {
public:
    AdvancedSimulation(PopType& pop, uint64_t seed,
                        SelectionPolicy sel = {}, MatingSystem mate = {});

    // Fluent configuration
    auto& set_events(SimulationEvents<PopType> ev);
    auto& set_demographics(DemographicSchedule sched);
    auto& enable_ancestry(bool flag = true);
    auto& set_gc_interval(Generation interval);

    void run(Generation num_generations);
    void step(Generation g);

    Generation generation() const;
    SelectionPolicy& selection_policy();
    MatingSystem& mating_system();
    mt19937_64& rng();
    double fitness_offset() const;
    const std::vector<Substitution>& substitutions() const;
    AncestryTracker* ancestry();
    SimulationEvents<PopType>& events();
};
```

**Generation loop (advanced):**
1. `on_generation_start` hook
2. `compute_fitness()` → `on_post_fitness`
3. `on_pre_selection_mut` (mutable hook)
4. For each offspring: `mating_system_(inds, fit, rng)` → `ParentPair` → `recombine()` each
5. `on_post_recombination`
6. `mutate()` → `on_post_mutation`
7. `advance()` (swap buffers)
8. GC sweep (if interval hit)
9. `on_post_advance_mut` (mutable hook)
10. `on_generation_end`
11. `should_stop` check

---

## 18. Event / Hook System

**Header:** `events.hpp`

The hook system provides fine-grained callbacks at every phase of the generation loop. All callbacks are `std::function` and nullable (zero overhead when not set).

```cpp
template <typename PopType>
struct SimulationEvents {
    // Read-only hooks
    std::function<void(Generation, const PopType&)> on_generation_start;
    std::function<void(Generation, const PopType&)> on_post_fitness;
    std::function<void(Generation, const PopType&)> on_post_recombination;
    std::function<void(Generation, const PopType&)> on_post_mutation;
    std::function<void(Generation, const PopType&)> on_generation_end;

    // Mutable hooks (can modify population)
    std::function<void(Generation, PopType&)> on_pre_selection_mut;
    std::function<void(Generation, PopType&)> on_post_advance_mut;

    // Early-stop predicate
    std::function<bool(Generation, const PopType&)> should_stop;

    bool empty() const;
};
```

### DataRecorder

Pre-built listener that accumulates per-generation measurements.

```cpp
template <typename PopType>
struct DataRecorder {
    bool track_mean_fitness = true;
    bool track_segregating_sites = true;
    Generation sample_interval = 1;

    std::vector<Generation> generations;
    std::vector<double>     mean_fitness;
    std::vector<size_t>     segregating_sites;

    auto as_callback();   // returns a function for on_generation_end
    void clear();
};
```

### Example: custom hooks

```cpp
SimulationEvents<Pop> events;

events.on_generation_end = [](Generation g, const Pop& p) {
    if ((g + 1) % 10 == 0)
        std::cout << "gen " << g + 1 << "  w=" << p.mean_fitness() << "\n";
};

events.should_stop = [](Generation g, const Pop& p) {
    return p.segregating_sites() == 0;  // stop when monomorphic
};

// Mutable hook: inject migrants every 50 generations
events.on_post_advance_mut = [&](Generation g, Pop& p) {
    if ((g + 1) % 50 == 0) inject_migrants(p);
};

sim.set_events(std::move(events));
```

---

## 19. Demographics

**Header:** `demographics.hpp`

Models time-varying population size.

### Event types

| Type | Fields | Description |
|------|--------|-------------|
| `SizeChangeEvent` | `when`, `new_N` | Instantaneous size change at generation `when` |
| `ExponentialGrowthEvent` | `when`, `rate`, `base_N` | N(t) = base·exp(rate·(t−when)). `base_N=0` uses current N. |
| `BottleneckEvent` | `when`, `bottleneck_N`, `duration`, `restore_N` | Reduce to `bottleneck_N` for `duration` gens, then restore |

### DemographicSchedule

```cpp
class DemographicSchedule {
public:
    void add_size_change(Generation when, size_t new_N);
    void add_exponential_growth(Generation when, double rate, size_t base_N = 0);
    void add_bottleneck(Generation when, size_t bottleneck_N,
                        Generation duration, size_t restore_N);
    size_t population_size(Generation gen, size_t current_N) const;
    bool empty() const;
};
```

Events are kept sorted by `when`. The `population_size()` method returns at least 1.

### Example: bottleneck + recovery

```cpp
DemographicSchedule sched;
sched.add_bottleneck(50, /*bottleneck*/20, /*duration*/10, /*restore*/500);
sched.add_exponential_growth(100, /*rate*/0.05);

sim.set_demographics(std::move(sched));
```

---

## 20. Migration & Demes

**Header:** `migration.hpp`

### DemeAssignment

```cpp
class DemeAssignment {
public:
    DemeAssignment(size_t N, size_t num_demes);       // equal-size demes
    DemeAssignment(std::vector<DemeID> assignments);  // explicit

    DemeID deme_of(size_t individual) const;
    size_t num_demes() const;
    const std::vector<size_t>& individuals_in(DemeID d) const;
    size_t deme_size(DemeID d) const;
    size_t total_size() const;
    void set_deme(size_t individual, DemeID new_deme);
    void rebuild_index();
};
```

### MigrationMatrix

```cpp
class MigrationMatrix {
public:
    MigrationMatrix(size_t num_demes);           // identity (no migration)
    static MigrationMatrix island_model(size_t num_demes, double m);

    void set(DemeID target, DemeID source, double prob);
    double get(DemeID target, DemeID source) const;
    DemeID sample_source_deme(DemeID target, mt19937_64& rng) const;
    bool is_valid(double tol = 1e-6) const;      // rows sum to ~1
};
```

The `island_model()` builder creates a symmetric matrix: M[i][i] = 1−m, M[i][j] = m/(D−1).

### Example: 3-deme island model

```cpp
auto demes = DemeAssignment(N, 3);
auto mig = MigrationMatrix::island_model(3, 0.05);  // 5% migration

// In manual sim loop: for each offspring in deme d,
DemeID source = mig.sample_source_deme(d, rng);
// then pick a parent from demes.individuals_in(source)
```

---

## 21. Statistics

**Header:** `statistics.hpp`

### SummaryStats

```cpp
struct SummaryStats {
    size_t num_haplotypes;
    size_t num_sites;
    size_t segregating_sites;
    double pi;                  // nucleotide diversity
    double theta_w;             // Watterson's theta
    double tajimas_d;           // Tajima's D
    double mean_heterozygosity;
    std::vector<size_t> sfs;    // site frequency spectrum
};
```

### Functions

| Function | Description |
|----------|-------------|
| `compute_sfs_dense(storage)` | Unfolded SFS. sfs[k] = sites with k non-ref copies. |
| `compute_sfs_sparse(storage)` | Same, for SparseVariants |
| `fold_sfs(sfs)` | Minor allele frequency spectrum |
| `compute_pi_from_sfs(sfs)` | Nucleotide diversity from SFS |
| `compute_theta_w(S, n)` | Watterson's θ = S / a₁ |
| `compute_tajimas_d(pi, theta_w, S, n)` | Full Tajima 1989 formula |
| `site_heterozygosity_dense/sparse(storage, site)` | 1 − Σp² at site |
| `mean_heterozygosity_dense/sparse(storage)` | Mean H across all sites |
| `compute_summary_dense/sparse(storage)` | All-in-one → `SummaryStats` |
| `print_sfs(sfs, max_k)` | Print non-zero SFS bins to stdout |

### Generic versions (concept-constrained)

```cpp
template <ElementAccessStorage Storage>
auto compute_sfs(const Storage& s) → vector<size_t>;

template <ElementAccessStorage Storage>
auto compute_summary(const Storage& s) → SummaryStats;
```

These auto-dispatch to the optimised Dense or Sparse implementation at compile time.

---

## 22. Linkage Disequilibrium

**Header:** `ld.hpp`

### Types

```cpp
struct PairwiseLD { SiteIndex site_i, site_j; double D, D_prime, r_sq; };
struct LDDecayPoint { size_t distance; double mean_r_sq; size_t n_pairs; };
```

### Functions

| Function | Description |
|----------|-------------|
| `compute_ld<S>(storage, i, j)` | D, D′, r² between two sites |
| `compute_ld_decay<S>(storage, bin_size, start, end, max_pairs)` | Mean r² vs physical distance, binned |
| `compute_ld_matrix<S>(storage, start, end)` | All pairwise r² in [start, end), upper triangle |

`compute_ld` auto-dispatches to a `hap_ptr`-based fast path for Dense storage.

---

## 23. Population Differentiation (Fst)

**Header:** `fst.hpp`

All functions require `ElementAccessStorage`.

| Function | Description |
|----------|-------------|
| `nei_gst_site<S>(storage, site, demes)` | Nei's Gst = 1 − H_s/H_t per site |
| `nei_gst<S>(storage, demes)` | Genome-wide mean Gst |
| `hudson_fst_site<S>(storage, site, deme_a, deme_b)` | Hudson's Fst per site |
| `hudson_fst<S>(storage, deme_a, deme_b)` | Genome-wide ratio-of-averages |
| `pairwise_fst_matrix<S>(storage, demes)` | D×D symmetric matrix |
| `individual_fis<S>(storage, individual)` | 1 − H_o (observed heterozygosity) |
| `mean_fis<S>(storage)` | Mean inbreeding coefficient |

---

## 24. Garbage Collection

**Header:** `gc.hpp`

Scans the storage for alleles that have either been **fixed** (all haplotypes carry the same non-ref allele) or **lost** (all haplotypes are reference). Fixed alleles are recorded as substitutions and their fitness contribution is accumulated into an offset.

### Types

```cpp
struct Substitution { AlleleID allele_id; SiteIndex site; Generation fixation_gen; };
struct GCResult    { size_t lost_count; size_t fixed_count;
                     double fitness_offset; std::vector<Substitution> substitutions; };
```

### Functions

```cpp
GCResult gc_sweep_dense(DenseHaplotypes& s, const AlleleTable& t, Generation g);
GCResult gc_sweep_sparse(SparseVariants& s, const AlleleTable& t, Generation g);

// Generic (concept-constrained):
template <HaplotypeStorage S>
GCResult gc_sweep(S& s, const AlleleTable& t, Generation g);
```

**Integration with AdvancedSimulation**: Enable via `sim.set_gc_interval(20)`. The driver accumulates `fitness_offset()` and `substitutions()`.

---

## 25. Ancestry Tracking

**Header:** `ancestry.hpp`

Records a pedigree (parent–offspring relationship) for every individual born. Opt-in via `sim.enable_ancestry()`.

### Types

```cpp
struct IndividualID   { Generation gen; size_t index; };
struct BirthRecord    { IndividualID offspring, parent0, parent1;
                        std::vector<SiteIndex> breakpoints0, breakpoints1; };
```

### AncestryTracker API

```cpp
class AncestryTracker {
public:
    AncestryTracker(size_t reserve_hint = 0);

    void begin_generation(Generation g);
    void record_birth(size_t offspring_idx, size_t parent0_idx,
                      size_t parent1_idx,
                      std::vector<SiteIndex> bp0 = {}, std::vector<SiteIndex> bp1 = {});

    const std::vector<BirthRecord>& records() const;
    std::vector<BirthRecord> offspring_of_generation(Generation g) const;
    std::optional<BirthRecord> find(IndividualID id) const;
    std::vector<IndividualID> trace_lineage(IndividualID start,
                                             size_t max_depth = 100) const;
    size_t effective_breeders(Generation offspring_gen) const;
    void prune_before(Generation keep_from);
    void clear();
    size_t size() const;
};
```

**Memory**: O(N·G) for full pedigree. Use `prune_before()` to manage memory in long runs.

---

## 26. Serialization / Checkpointing

**Header:** `serialization.hpp`

Binary checkpoint / restore for simulation state. Architecture-specific format (not portable across platforms).

### Checkpoint format

```
[CheckpointHeader: magic(4) version(4) gen(8) backend(4)]
[AlleleTable: count(8) + entries×(gen + sel_coeff + dominance + label)]
[Storage data: Dense(row-by-row) or Sparse(length-prefixed variant list)]
[RNG state: mt19937_64 << operator]
```

### Functions

```cpp
// Save
void save_checkpoint(const std::string& path, Generation gen,
                     const DenseHaplotypes& storage,
                     const AlleleTable& table, const mt19937_64& rng);
void save_checkpoint(const std::string& path, Generation gen,
                     const SparseVariants& storage,
                     const AlleleTable& table, const mt19937_64& rng);

// Inspect
CheckpointHeader read_checkpoint_header(const std::string& path);

// Load
struct DenseCheckpoint  { Generation gen; DenseHaplotypes storage; mt19937_64 rng; };
struct SparseCheckpoint { Generation gen; SparseVariants  storage; mt19937_64 rng; };

DenseCheckpoint  load_checkpoint_dense(const std::string& path, AlleleTable& table);
SparseCheckpoint load_checkpoint_sparse(const std::string& path, AlleleTable& table);
```

### Example: checkpoint every 100 generations

```cpp
events.on_generation_end = [&](Generation g, const Pop& p) {
    if ((g + 1) % 100 == 0)
        save_checkpoint("sim_gen" + std::to_string(g+1) + ".ckpt",
                        g+1, p.storage(), p.alleles(), sim.rng());
};
```

---

## 27. Output Formats

**Header:** `output_formats.hpp`

Export simulation data in standard formats for downstream analysis with external tools.

### VCF (Variant Call Format v4.2)

```cpp
struct VCFOptions {
    std::string contig_name = "chr1";
    std::string source      = "gensim";
    bool        phased      = true;
    size_t      ploidy      = 2;
};

void write_vcf(std::ostream& out, const DenseHaplotypes& storage,
               const AlleleTable& table, const VCFOptions& opts = {},
               const std::vector<std::string>& sample_names = {});
bool write_vcf_file(const std::string& filename, ...);
```

Only segregating sites are written. Multi-allelic sites are fully supported. Compatible with **bcftools**, **PLINK**, **GATK**.

### Hudson's ms format

```cpp
void write_ms(std::ostream& out, const DenseHaplotypes& storage,
              const std::string& command_line = "gensim");
bool write_ms_file(const std::string& filename, ...);
```

Biallelic (0/1) format. Positions normalised to [0, 1]. Compatible with **ms**, **msprime**, analysis scripts.

### PLINK PED/MAP

```cpp
void write_ped(std::ostream& out, const DenseHaplotypes& storage,
               const std::vector<Individual>& inds = {},
               const std::string& family_id = "FAM1");
void write_map(std::ostream& out, size_t num_sites, const std::string& chrom = "1");
bool write_plink(const std::string& basename, const DenseHaplotypes& storage,
                  const std::vector<Individual>& inds = {});
```

Writes `.ped` and `.map` files. Alleles encoded as 1 (ref) / 2 (alt). Sex is encoded from `Individual::sex`.

### EIGENSTRAT (.geno/.snp/.ind)

```cpp
void write_eigenstrat(const std::string& basename, const DenseHaplotypes& storage,
                       const std::vector<Individual>& inds = {});
```

Writes three files: `.geno` (0/1/2 dosage), `.snp` (SNP info), `.ind` (individual info). Compatible with **SmartPCA**, **EIGENSOFT**.

### Genotype matrix (tab-separated)

```cpp
void write_genotype_matrix(std::ostream& out, const DenseHaplotypes& storage);
```

Rows = individuals, columns = sites. Values = number of non-ref alleles (0, 1, 2). For quick import into R/Python.

---

## 28. Writing Custom Policies

All custom policies just need to satisfy the relevant concept. No base class, no registration — if it compiles with the concept, it works.

### Custom storage backend

```cpp
class MyStorage {
public:
    size_t num_individuals() const;
    size_t num_sites() const;
    size_t num_haplotypes() const;
    void copy_segment(size_t src, size_t dst, SiteIndex begin, SiteIndex end);
    void swap_buffers();
    size_t count_segregating_sites() const;

    // For ElementAccessStorage (needed by statistics, LD, Fst):
    AlleleID get(size_t hap, SiteIndex site) const;
    AlleleID offspring_get(size_t hap, SiteIndex site) const;
    void offspring_set(size_t hap, SiteIndex site, AlleleID a);
};
static_assert(gensim::ElementAccessStorage<MyStorage>);  // compile-time check
```

### Custom mutation policy

```cpp
struct MyMutation {
    void operator()(DenseHaplotypes& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const {
        // Add mutations to the back (offspring) buffer
        // Use storage.offspring_set() or storage.offspring_hap_ptr()
    }

    // Optionally provide SparseVariants overload
    void operator()(SparseVariants& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const { ... }
};
```

### Custom fitness policy

```cpp
struct EpistaticFitness {
    Fitness operator()(const DenseHaplotypes& storage, size_t ind,
                       const AlleleTable& table) const {
        const size_t L = storage.num_sites();
        const AlleleID* h0 = storage.hap_ptr(2 * ind);
        const AlleleID* h1 = storage.hap_ptr(2 * ind + 1);

        Fitness w = 1.0;
        // Check for epistatic interactions between sites...
        for (size_t i = 0; i < L; ++i) {
            for (size_t j = i + 1; j < L; ++j) {
                if (h0[i] != kRefAllele && h0[j] != kRefAllele)
                    w *= 0.95;  // negative epistasis
            }
        }
        return std::max(w, 0.0);
    }
};
```

### Custom mating system

```cpp
struct AgeDependentMating {
    ParentPair operator()(const std::vector<Individual>& inds,
                           const std::vector<Fitness>& fit,
                           std::mt19937_64& rng) const {
        // Weight by fitness × age-dependent fecundity
        std::vector<double> weights(inds.size());
        for (size_t i = 0; i < inds.size(); ++i) {
            double fecundity = (inds[i].age >= 2 && inds[i].age <= 10) ? 1.0 : 0.1;
            weights[i] = fit[i] * fecundity;
        }
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        return {dist(rng), dist(rng)};
    }
};
static_assert(gensim::MatingSystemConcept<AgeDependentMating>);
```

### Custom selection policy

```cpp
struct FrequencyDependentSelection {
    double operator()(const std::vector<Fitness>& fitnesses,
                       std::mt19937_64& rng) const {
        // ... rare-type advantage logic ...
    }
};
static_assert(gensim::SelectionPolicyConcept<FrequencyDependentSelection>);
```

---

## 29. Performance Considerations

### Memory layout

| Backend | Memory | Access pattern |
|---------|--------|---------------|
| DenseHaplotypes | 16NL bytes (2 buffers) | Row-major, cache-friendly for per-haplotype scans |
| SparseVariants | ∝ V (segregating variants) | Binary search per access, but much less memory when V ≪ L |

### Mutation performance

All mutation policies use Poisson-scattered placement:
- Draw `k ~ Poisson(μ·L)` mutations per haplotype
- Place each at a random site
- Complexity: **O(μ·L·2N)** instead of O(L·2N)
- This is a huge win when μ·L is small (typical for realistic parameters)

### Recombination performance

- `SingleCrossover` on Dense uses `std::memcpy` for the two segments — very fast
- `MultiBreakpoint` sorts breakpoints and copies alternating segments
- `MappedCrossover` uses cumulative binary search for weighted breakpoint placement

### Selection performance

`FitnessProportionate` constructs a `std::discrete_distribution` per call. For the basic `Simulation` driver, this is precomputed once per generation (cached). For individual policies called per-offspring, consider caching the distribution externally.

### GC scheduling

Without GC, the `AlleleTable` grows monotonically. Enable periodic GC via `sim.set_gc_interval(interval)`. Typical intervals: 10–50 generations. GC cost is proportional to L (Dense) or V (Sparse).

### `thread_local` usage

`SexualMating` and `AssortativeMating` use `thread_local` vectors to avoid per-call heap allocation. This is safe in single-threaded code and efficient in multi-threaded scenarios (each thread gets its own vectors).

---

## 30. Thread Safety

The library is **single-threaded by default**. Here is a guide for parallelisation:

| Component | Thread safety | Notes |
|-----------|--------------|-------|
| `AlleleTable` | **Not thread-safe** | Use mutex or per-thread tables + merge |
| `DenseHaplotypes` | Read-only front buffer is safe | Offspring buffer writes must be coordinated |
| `SparseVariants` | Read-only front buffer is safe | Same as Dense |
| Fitness evaluation | **Trivially parallel** | Read-only access to front buffer |
| Offspring production | **Embarrassingly parallel** per individual | Each offspring writes to distinct back-buffer rows. Each thread needs its own RNG. |
| Mutation | **Per-haplotype independent** | Parallel if AlleleTable is thread-safe |
| Statistics | **Read-only** | Safe to run in parallel |

### Recommended parallelisation strategy

```
// Per-thread RNG
std::vector<std::mt19937_64> thread_rngs;
for (int t = 0; t < num_threads; ++t)
    thread_rngs.emplace_back(base_seed + t);

// Partition individuals across threads
// Each thread:
//   1. Select parents (read-only fitness cache)
//   2. Recombine into back buffer rows [start, end)
//   3. Mutate back buffer rows [start, end) — need local AlleleTable
// Merge local AlleleTable mutations into global table
```

---

## 31. Extending the Library

### Adding a new policy type

1. Define the concept in `concepts.hpp` (or in the relevant header)
2. Add the new template parameter to `Population` or `AdvancedSimulation`
3. Provide default implementations
4. Add concept-based `static_assert` for early error detection

### Adding a new statistic

1. Write a function in `statistics.hpp` (or a new header)
2. Use `requires ElementAccessStorage<Storage>` for generic access
3. Provide optimised Dense and Sparse specialisations via `if constexpr (std::same_as<Storage, DenseHaplotypes>)`

### Adding a new output format

1. Add a function to `output_formats.hpp`
2. Take `const DenseHaplotypes&` (and optionally `const std::vector<Individual>&`)
3. Use `storage.get(hap, site)` for generic access or `storage.hap_ptr(hap)` for Dense fast path

### Adding a new storage backend

1. Create a class satisfying `HaplotypeStorage` (minimum) or `ElementAccessStorage` (recommended)
2. Verify with `static_assert(gensim::ElementAccessStorage<MyBackend>)`
3. Add overloads in your mutation/recombination policies for the new backend
4. All generic statistics, LD, Fst functions will work automatically

---

## 32. Troubleshooting & Common Pitfalls

### Compilation errors

| Error | Cause | Fix |
|-------|-------|-----|
| "concept not satisfied" | Policy doesn't match concept signature | Check the concept's `requires` clause — is the signature exact? |
| "no matching function for call" | Missing Dense/Sparse overload | Add the overload for the other storage backend |
| C4293: shift count exceeds type width | Using `<<32` on 32-bit types | Use `1ULL << 32` or `static_cast<uint64_t>(1) << 32` |
| Linker error LNK2005 (multiple definitions) | Including `.hpp` in multiple TUs | All functions are `inline` — ensure you're not defining non-inline free functions |

### Runtime issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Memory blowup over time | AlleleTable growing without GC | Enable `sim.set_gc_interval(N)` |
| `assert` failure in SexualMating | No males or no females in population | Call `pop.assign_sex(0.5, rng)` before simulation |
| Negative fitness values | Large deleterious mutations without clamping | AdditiveFitness/MultiplicativeFitness already clamp to 0; verify DFE parameters |
| All individuals identical | No mutation (μ=0) or very small population | Check mutation rate; increase N or μ |
| Tajima's D = NaN | 0 segregating sites | Handle this edge case in your analysis code |

### Design conventions

- **Never** modify the front (parent) buffer during a generation. It is read-only.
- **Always** call `prepare_offspring_buffer()` before writing offspring (critical for `SparseVariants`).
- **Always** call `compute_fitness()` before selection in a custom loop.
- When using `AdvancedSimulation`, prefer hooks over modifying the step loop directly.
- For multi-population migration: share a single `AlleleTable` by reference.
- When checkpointing: save the RNG state along with the genotype data to ensure reproducibility.

---

## Appendix: Complete Header Listing

| Header | Primary exports | Lines |
|--------|----------------|-------|
| `types.hpp` | `AlleleID`, `Sex`, `ChromosomeType`, `Individual`, `Position2D` | ~90 |
| `concepts.hpp` | 8 C++20 concepts | ~140 |
| `allele_table.hpp` | `AlleleInfo`, `AlleleTable` | ~60 |
| `dense_haplotypes.hpp` | `DenseHaplotypes` | ~180 |
| `sparse_variants.hpp` | `Variant`, `SparseVariants` | ~200 |
| `dfe.hpp` | 6 DFE types, `DFEVariant`, `MixtureDFE` | ~180 |
| `mutation_model.hpp` | `BernoulliInfiniteAlleles`, `BernoulliFiniteAlleles`, `DFEMutation` | ~200 |
| `recombination_model.hpp` | `RecombinationMap`, `SingleCrossover`, `MultiBreakpoint`, `MappedCrossover` | ~250 |
| `fitness_model.hpp` | `NeutralFitness`, `AdditiveFitness`, `MultiplicativeFitness`, `StabilizingSelectionFitness` | ~200 |
| `selection.hpp` | `FitnessProportionate`, `TournamentSelection`, `TruncationSelection`, `RankSelection`, `BoltzmannSelection` | ~200 |
| `mating_system.hpp` | 7 mating policies, `ParentPair`, `MatingSystemConcept` | ~340 |
| `chromosome.hpp` | `ChromosomeSpec`, `GenomeSpec`, `MultiChromGenome` | ~340 |
| `quantitative_trait.hpp` | 3 trait models, 3 trait fitness functions, `MovingOptimumSchedule`, `FrequencyDependentFitness` | ~420 |
| `mutation_rate_map.hpp` | `MutationRateMap`, `MutationTypeMap`, `MappedMutation`, `RegionalDFEMutation` | ~280 |
| `output_formats.hpp` | VCF, MS, PLINK, EIGENSTRAT, genotype matrix exporters | ~280 |
| `population.hpp` | `Population<Storage, Mut, Rec, Fit>` | ~170 |
| `demographics.hpp` | `DemographicSchedule`, 3 event types | ~120 |
| `migration.hpp` | `DemeAssignment`, `MigrationMatrix` | ~170 |
| `events.hpp` | `SimulationEvents`, `DataRecorder` | ~120 |
| `simulation.hpp` | `Simulation`, `AdvancedSimulation` | ~310 |
| `statistics.hpp` | `SummaryStats`, SFS, π, θ_W, Tajima's D, heterozygosity | ~300 |
| `ld.hpp` | `PairwiseLD`, `LDDecayPoint`, `compute_ld`, `compute_ld_decay` | ~220 |
| `fst.hpp` | `nei_gst`, `hudson_fst`, `pairwise_fst_matrix`, `individual_fis` | ~180 |
| `gc.hpp` | `Substitution`, `GCResult`, `gc_sweep` | ~150 |
| `ancestry.hpp` | `AncestryTracker`, `BirthRecord`, `IndividualID` | ~180 |
| `serialization.hpp` | `save_checkpoint`, `load_checkpoint_dense/sparse` | ~250 |
| `gensim.hpp` | Single-include header | ~60 |
