// =============================================================================
// output_formats.hpp — Standard file format export for interoperability.
//
// Exports simulation genotype data in standard formats used by downstream
// population genetics analysis tools:
//
//   VCF (Variant Call Format)  — GATK, bcftools, PLINK, etc.
//   MS format                  — ms, msprime, analysis pipelines
//   PED / MAP                  — PLINK
//   EIGENSTRAT .geno/.snp/.ind — SmartPCA, EIGENSOFT
//
// All exporters work with DenseHaplotypes and the AlleleTable to translate
// allele IDs to biallelic genotypes (ref/alt).  Multi-allelic sites are
// handled where the format supports it.
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "types.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace gensim {

// ─────────────────────────────────────────────────────────────────────────────
// VCF Export
// ─────────────────────────────────────────────────────────────────────────────

/// Options for VCF output.
struct VCFOptions {
    std::string contig_name = "chr1";     // CHROM column
    std::string source      = "gensim";   // ##source header
    bool        phased      = true;       // use '|' (phased) vs '/' (unphased)
    std::size_t ploidy      = 2;          // always 2 for diploid
};

/// Write genotype data in VCF 4.2 format.
///
/// Only segregating sites (sites with at least one non-reference allele)
/// are written.  Each unique non-reference allele at a site becomes an
/// ALT allele.  Genotypes are encoded as 0 (ref) or 1..n (alt).
inline void write_vcf(std::ostream& out,
                       const DenseHaplotypes& storage,
                       const AlleleTable& table,
                       const VCFOptions& opts = {},
                       const std::vector<std::string>& sample_names = {})
{
    const std::size_t N = storage.num_individuals();
    const std::size_t L = storage.num_sites();
    const char sep = opts.phased ? '|' : '/';

    // ── Header ──────────────────────────────────────────────────────────────
    out << "##fileformat=VCFv4.2\n";
    out << "##source=" << opts.source << "\n";
    out << "##contig=<ID=" << opts.contig_name << ",length=" << L << ">\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";

    for (std::size_t i = 0; i < N; ++i) {
        if (i < sample_names.size())
            out << '\t' << sample_names[i];
        else
            out << "\tind" << i;
    }
    out << '\n';

    // ── Data lines (one per segregating site) ───────────────────────────────
    for (std::size_t site = 0; site < L; ++site) {
        // Collect unique non-ref alleles at this site.
        std::vector<AlleleID> alt_alleles;
        for (std::size_t i = 0; i < N; ++i) {
            AlleleID a0 = storage.get(2 * i, site);
            AlleleID a1 = storage.get(2 * i + 1, site);
            auto maybe_add = [&](AlleleID a) {
                if (a != kRefAllele) {
                    auto it = std::find(alt_alleles.begin(), alt_alleles.end(), a);
                    if (it == alt_alleles.end()) alt_alleles.push_back(a);
                }
            };
            maybe_add(a0);
            maybe_add(a1);
        }

        if (alt_alleles.empty()) continue;  // monomorphic ref — skip

        // CHROM POS ID REF ALT QUAL FILTER INFO FORMAT
        out << opts.contig_name << '\t'
            << (site + 1) << '\t'              // 1-based position
            << '.' << '\t'                     // ID
            << '0' << '\t';                    // REF = allele 0

        // ALT: comma-separated allele IDs (or labels if available).
        for (std::size_t a = 0; a < alt_alleles.size(); ++a) {
            if (a > 0) out << ',';
            const auto& info = table[alt_alleles[a]];
            if (!info.label.empty())
                out << info.label;
            else
                out << alt_alleles[a];
        }

        out << "\t.\t.\t.\tGT";

        // Genotype columns.
        for (std::size_t i = 0; i < N; ++i) {
            out << '\t';
            AlleleID a0 = storage.get(2 * i, site);
            AlleleID a1 = storage.get(2 * i + 1, site);

            auto allele_idx = [&](AlleleID a) -> int {
                if (a == kRefAllele) return 0;
                auto it = std::find(alt_alleles.begin(), alt_alleles.end(), a);
                if (it != alt_alleles.end())
                    return static_cast<int>(std::distance(alt_alleles.begin(), it)) + 1;
                return 0;
            };

            out << allele_idx(a0) << sep << allele_idx(a1);
        }
        out << '\n';
    }
}

/// Write VCF to a file.
inline bool write_vcf_file(const std::string& filename,
                            const DenseHaplotypes& storage,
                            const AlleleTable& table,
                            const VCFOptions& opts = {},
                            const std::vector<std::string>& sample_names = {})
{
    std::ofstream ofs(filename);
    if (!ofs) return false;
    write_vcf(ofs, storage, table, opts, sample_names);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// MS Format Export
// ─────────────────────────────────────────────────────────────────────────────
// The MS format (from Hudson's ms program) is widely used in population
// genetics for exchanging haplotype data.
//
// Format:
//   //
//   segsites: <S>
//   positions: <p1> <p2> ... <pS>
//   <haplotype 1: 0s and 1s>
//   <haplotype 2: 0s and 1s>
//   ...
//
// Sites are biallelic (0 = ref, 1 = any non-ref).
inline void write_ms(std::ostream& out,
                      const DenseHaplotypes& storage,
                      const std::string& command_line = "gensim")
{
    const std::size_t N = storage.num_individuals();
    const std::size_t H = storage.num_haplotypes();
    const std::size_t L = storage.num_sites();

    // Find segregating sites.
    std::vector<std::size_t> seg_sites;
    for (std::size_t s = 0; s < L; ++s) {
        bool has_alt = false;
        for (std::size_t h = 0; h < H && !has_alt; ++h) {
            if (storage.get(h, s) != kRefAllele) has_alt = true;
        }
        if (has_alt) seg_sites.push_back(s);
    }

    // Header.
    out << command_line << "\n";
    out << N * 2 << " 1\n\n";  // nsamples nreps
    out << "//\n";
    out << "segsites: " << seg_sites.size() << "\n";

    if (!seg_sites.empty()) {
        out << "positions:";
        for (auto s : seg_sites) {
            out << ' ' << std::fixed << std::setprecision(6)
                << (static_cast<double>(s) / static_cast<double>(L));
        }
        out << '\n';

        // Haplotype lines.
        for (std::size_t h = 0; h < H; ++h) {
            for (auto s : seg_sites) {
                AlleleID a = storage.get(h, s);
                out << (a != kRefAllele ? '1' : '0');
            }
            out << '\n';
        }
    }
}

/// Write MS format to a file.
inline bool write_ms_file(const std::string& filename,
                           const DenseHaplotypes& storage,
                           const std::string& command_line = "gensim")
{
    std::ofstream ofs(filename);
    if (!ofs) return false;
    write_ms(ofs, storage, command_line);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// PLINK PED/MAP Export
// ─────────────────────────────────────────────────────────────────────────────
// PED file: one line per individual
//   FID IID PAT MAT SEX PHENO G1_A1 G1_A2 G2_A1 G2_A2 ...
// MAP file: one line per marker
//   CHR SNP_ID GENDIST BP_POS

/// Write PLINK PED file.
inline void write_ped(std::ostream& out,
                       const DenseHaplotypes& storage,
                       const std::vector<Individual>& inds = {},
                       const std::string& family_id = "FAM1")
{
    const std::size_t N = storage.num_individuals();
    const std::size_t L = storage.num_sites();

    for (std::size_t i = 0; i < N; ++i) {
        // FID IID PAT MAT SEX PHENO
        out << family_id << ' '
            << "ind" << i << ' '
            << "0 0 ";   // PAT MAT unknown

        // Sex: 1=male, 2=female, 0=unknown
        int sex_code = 0;
        if (i < inds.size()) {
            if (inds[i].sex == Sex::Male) sex_code = 1;
            else if (inds[i].sex == Sex::Female) sex_code = 2;
        }
        out << sex_code << " -9";  // phenotype unknown

        // Genotypes: space-separated pairs.
        for (std::size_t s = 0; s < L; ++s) {
            AlleleID a0 = storage.get(2 * i, s);
            AlleleID a1 = storage.get(2 * i + 1, s);
            // Encode as 1 (ref) or 2 (alt) for PLINK.
            char c0 = (a0 == kRefAllele) ? '1' : '2';
            char c1 = (a1 == kRefAllele) ? '1' : '2';
            out << ' ' << c0 << ' ' << c1;
        }
        out << '\n';
    }
}

/// Write PLINK MAP file.
inline void write_map(std::ostream& out,
                       std::size_t num_sites,
                       const std::string& chrom = "1")
{
    for (std::size_t s = 0; s < num_sites; ++s) {
        out << chrom << '\t'
            << "snp" << s << '\t'
            << "0\t"              // genetic distance
            << (s + 1) << '\n';   // base-pair position
    }
}

/// Write paired PED and MAP files.
inline bool write_plink(const std::string& basename,
                         const DenseHaplotypes& storage,
                         const std::vector<Individual>& inds = {})
{
    std::ofstream ped(basename + ".ped");
    std::ofstream map(basename + ".map");
    if (!ped || !map) return false;
    write_ped(ped, storage, inds);
    write_map(map, storage.num_sites());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// EIGENSTRAT .geno/.snp/.ind Export
// ─────────────────────────────────────────────────────────────────────────────
// Used by SmartPCA and EIGENSOFT for PCA of population structure.
//
// .geno: N characters per line (one per individual), S lines (one per SNP)
//   0 = homozygous ref, 1 = heterozygous, 2 = homozygous alt, 9 = missing
// .snp: SNP info (one per line)
// .ind: individual info (one per line)

inline void write_eigenstrat(const std::string& basename,
                              const DenseHaplotypes& storage,
                              const std::vector<Individual>& inds = {})
{
    const std::size_t N = storage.num_individuals();
    const std::size_t L = storage.num_sites();

    // .geno file
    {
        std::ofstream geno(basename + ".geno");
        for (std::size_t s = 0; s < L; ++s) {
            for (std::size_t i = 0; i < N; ++i) {
                AlleleID a0 = storage.get(2 * i, s);
                AlleleID a1 = storage.get(2 * i + 1, s);
                int alt_count = (a0 != kRefAllele ? 1 : 0) +
                                (a1 != kRefAllele ? 1 : 0);
                geno << alt_count;
            }
            geno << '\n';
        }
    }

    // .snp file
    {
        std::ofstream snp(basename + ".snp");
        for (std::size_t s = 0; s < L; ++s) {
            snp << "snp" << s << "\t1\t0.0\t" << (s + 1) << "\t0\t1\n";
        }
    }

    // .ind file
    {
        std::ofstream ind(basename + ".ind");
        for (std::size_t i = 0; i < N; ++i) {
            std::string sex_str = "U";
            if (i < inds.size()) {
                if (inds[i].sex == Sex::Male) sex_str = "M";
                else if (inds[i].sex == Sex::Female) sex_str = "F";
            }
            ind << "ind" << i << "\t" << sex_str << "\tPop1\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Genotype matrix export (plain text, for quick analysis).
// ─────────────────────────────────────────────────────────────────────────────
// Rows = individuals, columns = sites.
// Values = number of non-reference alleles (0, 1, or 2).
inline void write_genotype_matrix(std::ostream& out,
                                   const DenseHaplotypes& storage)
{
    const std::size_t N = storage.num_individuals();
    const std::size_t L = storage.num_sites();

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t s = 0; s < L; ++s) {
            if (s > 0) out << '\t';
            AlleleID a0 = storage.get(2 * i, s);
            AlleleID a1 = storage.get(2 * i + 1, s);
            int count = (a0 != kRefAllele ? 1 : 0) +
                        (a1 != kRefAllele ? 1 : 0);
            out << count;
        }
        out << '\n';
    }
}

}  // namespace gensim
