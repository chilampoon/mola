# mola

### Multi-modal Observations from Long- and short-read Alignments


`mola` links single-nucleotide mismatches to read alignments from all kinds of RNA sequencing data:
- bulk short-read paired-/single-end RNA-seq
- bulk long-read RNA-seq
- single-cell short-read RNA-seq
- single-cell long-read RNA-seq

then implement probabilistic modeling for:
- classification of sequencing errors, RNA editing and germline SNPs
- haplotype phasing
- somatic mutation detection

## Installation

Install from source for now:

```bash
git clone git@github.com:chilampoon/mola.git
cd mola
pip install -e .
```


## Use Cases

| Application | Data modality | Publication |
|---|---|---|
| A-to-I editing site calling from HyperTRIBE | Bulk short-read RNA-seq | [Agarwal et al. 2026](https://doi.org/10.1126/science.adx4174) |

## Documentation
Long-form documentation lives in the Fumadocs app under `docs/`.
