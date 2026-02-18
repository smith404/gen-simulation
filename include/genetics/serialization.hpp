// =============================================================================
// serialization.hpp — Checkpoint / restore for simulations.
//
// Save and load simulation state to/from binary files so that long-running
// simulations can be:
//   - Paused and resumed
//   - Forked (save, then run two variants from the same checkpoint)
//   - Reproduced deterministically (save state + RNG)
//
// Format is a simple binary dump — not portable across architectures or
// compiler versions (due to std::mt19937_64 state).  For portable I/O,
// users should implement VCF, PLINK, or tree-sequence output.
//
// Supported:
//   - DenseHaplotypes (bulk binary)
//   - SparseVariants (length-prefixed variant list)
//   - AlleleTable (entry-by-entry)
//   - RNG state (std::mt19937_64 via operator<</>>)
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gensim {

// ── Magic number and version for file identification ────────────────────────
inline constexpr std::uint32_t kCheckpointMagic   = 0x47454E53;  // "GENS"
inline constexpr std::uint32_t kCheckpointVersion = 1;

// ── Low-level binary I/O helpers ────────────────────────────────────────────

namespace detail {

template <typename T>
void write_pod(std::ostream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
void read_pod(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

inline void write_string(std::ostream& os, const std::string& s) {
    std::uint32_t len = static_cast<std::uint32_t>(s.size());
    write_pod(os, len);
    os.write(s.data(), len);
}

inline std::string read_string(std::istream& is) {
    std::uint32_t len;
    read_pod(is, len);
    std::string s(len, '\0');
    is.read(s.data(), len);
    return s;
}

}  // namespace detail

// ── AlleleTable serialization ───────────────────────────────────────────────

inline void save_allele_table(std::ostream& os, const AlleleTable& table) {
    std::uint64_t n = table.size();
    detail::write_pod(os, n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& info = table[static_cast<AlleleID>(i)];
        detail::write_pod(os, info.origin_gen);
        detail::write_pod(os, info.sel_coeff);
        detail::write_pod(os, info.dominance_h);
        detail::write_string(os, info.label);
    }
}

inline void load_allele_table(std::istream& is, AlleleTable& table) {
    // The table starts with ID 0 (ref) already inserted.
    // We overwrite it and add the rest.
    std::uint64_t n;
    detail::read_pod(is, n);
    for (std::uint64_t i = 0; i < n; ++i) {
        AlleleInfo info;
        detail::read_pod(is, info.origin_gen);
        detail::read_pod(is, info.sel_coeff);
        detail::read_pod(is, info.dominance_h);
        info.label = detail::read_string(is);
        if (i == 0) {
            // Overwrite the ref allele.
            table[kRefAllele] = info;
        } else {
            table.new_allele(std::move(info));
        }
    }
}

// ── DenseHaplotypes serialization ───────────────────────────────────────────

inline void save_dense(std::ostream& os, const DenseHaplotypes& storage) {
    std::uint64_t N = storage.num_individuals();
    std::uint64_t L = storage.num_sites();
    detail::write_pod(os, N);
    detail::write_pod(os, L);

    // Write front buffer (current generation) row by row.
    const std::size_t H = storage.num_haplotypes();
    for (std::size_t h = 0; h < H; ++h) {
        os.write(reinterpret_cast<const char*>(storage.hap_ptr(h)),
                 static_cast<std::streamsize>(L * sizeof(AlleleID)));
    }
}

inline DenseHaplotypes load_dense(std::istream& is) {
    std::uint64_t N, L;
    detail::read_pod(is, N);
    detail::read_pod(is, L);

    DenseHaplotypes storage(static_cast<std::size_t>(N),
                             static_cast<std::size_t>(L));
    const std::size_t H = 2 * static_cast<std::size_t>(N);
    for (std::size_t h = 0; h < H; ++h) {
        is.read(reinterpret_cast<char*>(storage.hap_ptr(h)),
                static_cast<std::streamsize>(L * sizeof(AlleleID)));
    }
    return storage;
}

// ── SparseVariants serialization ────────────────────────────────────────────

inline void save_sparse(std::ostream& os, const SparseVariants& storage) {
    std::uint64_t N = storage.num_individuals();
    std::uint64_t L = storage.num_sites();
    detail::write_pod(os, N);
    detail::write_pod(os, L);

    const auto& vars = storage.variants();
    std::uint64_t nvar = vars.size();
    detail::write_pod(os, nvar);

    for (auto& v : vars) {
        detail::write_pod(os, v.pos);
        // Write alleles vector.
        std::uint64_t na = v.alleles.size();
        detail::write_pod(os, na);
        os.write(reinterpret_cast<const char*>(v.alleles.data()),
                 static_cast<std::streamsize>(na * sizeof(AlleleID)));
    }
}

inline SparseVariants load_sparse(std::istream& is) {
    std::uint64_t N, L;
    detail::read_pod(is, N);
    detail::read_pod(is, L);

    SparseVariants storage(static_cast<std::size_t>(N),
                            static_cast<std::size_t>(L));

    std::uint64_t nvar;
    detail::read_pod(is, nvar);

    auto& vars = storage.variants();
    vars.resize(static_cast<std::size_t>(nvar));
    for (auto& v : vars) {
        detail::read_pod(is, v.pos);
        std::uint64_t na;
        detail::read_pod(is, na);
        v.alleles.resize(static_cast<std::size_t>(na));
        is.read(reinterpret_cast<char*>(v.alleles.data()),
                static_cast<std::streamsize>(na * sizeof(AlleleID)));
    }
    return storage;
}

// ── RNG state serialization ─────────────────────────────────────────────────

inline void save_rng(std::ostream& os, const std::mt19937_64& rng) {
    os << rng;
}

inline void load_rng(std::istream& is, std::mt19937_64& rng) {
    is >> rng;
}

// ── Full checkpoint (Dense backend) ─────────────────────────────────────────

struct CheckpointHeader {
    std::uint32_t magic   = kCheckpointMagic;
    std::uint32_t version = kCheckpointVersion;
    Generation    gen     = 0;
    std::uint8_t  backend = 0;   // 0 = Dense, 1 = Sparse
};

/// Save a complete simulation checkpoint to a binary file.
/// Backend: 0 = Dense, 1 = Sparse.
inline void save_checkpoint(const std::string& path,
                              Generation gen,
                              const DenseHaplotypes& storage,
                              const AlleleTable& table,
                              const std::mt19937_64& rng)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open checkpoint file for writing: " + path);

    CheckpointHeader hdr;
    hdr.gen     = gen;
    hdr.backend = 0;
    detail::write_pod(ofs, hdr);
    save_allele_table(ofs, table);
    save_dense(ofs, storage);
    save_rng(ofs, rng);
}

/// Save a complete simulation checkpoint (Sparse backend).
inline void save_checkpoint(const std::string& path,
                              Generation gen,
                              const SparseVariants& storage,
                              const AlleleTable& table,
                              const std::mt19937_64& rng)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open checkpoint file for writing: " + path);

    CheckpointHeader hdr;
    hdr.gen     = gen;
    hdr.backend = 1;
    detail::write_pod(ofs, hdr);
    save_allele_table(ofs, table);
    save_sparse(ofs, storage);
    save_rng(ofs, rng);
}

/// Read the checkpoint header to determine backend type and generation.
inline CheckpointHeader read_checkpoint_header(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open checkpoint file: " + path);

    CheckpointHeader hdr;
    detail::read_pod(ifs, hdr);
    if (hdr.magic != kCheckpointMagic)
        throw std::runtime_error("Invalid checkpoint file (bad magic)");
    if (hdr.version != kCheckpointVersion)
        throw std::runtime_error("Unsupported checkpoint version");
    return hdr;
}

/// Load a Dense checkpoint.  Returns {generation, storage, rng_state}.
struct DenseCheckpoint {
    Generation      gen;
    DenseHaplotypes storage;
    std::mt19937_64 rng;
};

inline DenseCheckpoint load_checkpoint_dense(const std::string& path,
                                               AlleleTable& table)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open checkpoint file: " + path);

    CheckpointHeader hdr;
    detail::read_pod(ifs, hdr);
    if (hdr.magic != kCheckpointMagic)
        throw std::runtime_error("Invalid checkpoint file");
    if (hdr.backend != 0)
        throw std::runtime_error("Checkpoint is not Dense backend");

    load_allele_table(ifs, table);
    auto storage = load_dense(ifs);
    std::mt19937_64 rng;
    load_rng(ifs, rng);

    return DenseCheckpoint{hdr.gen, std::move(storage), rng};
}

/// Load a Sparse checkpoint.
struct SparseCheckpoint {
    Generation      gen;
    SparseVariants  storage;
    std::mt19937_64 rng;
};

inline SparseCheckpoint load_checkpoint_sparse(const std::string& path,
                                                  AlleleTable& table)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open checkpoint file: " + path);

    CheckpointHeader hdr;
    detail::read_pod(ifs, hdr);
    if (hdr.magic != kCheckpointMagic)
        throw std::runtime_error("Invalid checkpoint file");
    if (hdr.backend != 1)
        throw std::runtime_error("Checkpoint is not Sparse backend");

    load_allele_table(ifs, table);
    auto storage = load_sparse(ifs);
    std::mt19937_64 rng;
    load_rng(ifs, rng);

    return SparseCheckpoint{hdr.gen, std::move(storage), rng};
}

}  // namespace gensim
