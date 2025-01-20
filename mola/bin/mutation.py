import click
import os, gzip
import logging
from mola.mutation.mapping import map_read_mismatch
from mola.mutation.mismatch_site import *

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@click.group('mut')
def mola_mut():
    '''
    Mutation detection and analysis
    '''
    pass


@mola_mut.command('map')
@click.option('-b', '--bam', type=click.Path(exists=True), required=True,
              help="BAM/SAM file")
@click.option('--pileup_vcf', type=click.Path(exists=True), required=True,
              help="VCF output from pileup")
@click.option('--ref_vcf', type=click.Path(exists=True), required=False, default=None,
              help="bgzipped VCF file with tabix index, e.g. dbSNP vcf")
@click.option('-r', '--reads_dir', type=click.Path(exists=True), required=False,
              help="Directory storing Read objects, output from read annotate")
@click.option('--stranded', is_flag=True, show_default=True, default=False,
              help="Reads are from a stranded protocol")
@click.option('--paired_end', is_flag=True, show_default=True, default=False,
              help="Reads are from paired-end sequencing")
@click.option('--min_depth', show_default=True, default=10,
              help="Minimal pseudobulk depth for a mismatch")
@click.option('--min_minor_allele', show_default=True, default=1,
              help="Minimal pseudobulk minor allele count for a mismatch")
@click.option('--not_startswith_chr', is_flag=True, show_default=True, default=False,
              help="If specify, all chromosome ids must not start with 'chr'")
@click.option('-t', '--num_threads', type=int, default=None,
              help="Number of threads for parallel processing")
@click.option('--tmp_dir', type=click.Path(exists=True), required=True,
              help="Tmp directory from read annotation")
@click.option('-o', '--out_dir', default=os.getcwd(),
              help="Output directory, default is current working directory")
def read2mismatch(bam, pileup_vcf, ref_vcf, reads_dir, stranded, paired_end, min_depth,
                  min_minor_allele, not_startswith_chr, num_threads, out_dir, tmp_dir):
    '''Mappings between reads and mismatches'''
    os.makedirs(out_dir, exist_ok=True)
    startswith_chr = not not_startswith_chr
    map_read_mismatch(
        bam,
        pileup_vcf,
        ref_vcf,
        stranded,
        paired_end,
        min_depth,
        min_minor_allele,
        startswith_chr,
        num_threads,
        reads_dir,
        out_dir,
        tmp_dir
    )


@mola_mut.command('write')
@click.option('--site_dir', type=click.Path(exists=True), required=True,
              help="sites directory from mut map outputs")
@click.option('--stranded', is_flag=True, show_default=True, default=False,
              help="Output stranded site or not")
@click.option('--mode', show_default=True, default='bulk', 
              help="Output mode: bulk, pseudobulk, cell")
@click.option('--celltype_map', type=click.Path(exists=True), show_default=True, default=None,
              help="Cell type mapping file, first column is barcode, second is cell type")
@click.option('--matrix', is_flag=True, show_default=True, default=False,
              help="Output cell by site matrix instead of long table")
@click.option('-o', '--out_dir', default=os.getcwd(),
              help="Output directory, default is current working directory")
def write_site_table(site_dir, stranded, mode, celltype_map, matrix, out_dir):
    '''Write mismatch site table at bulk/pseudobulk/cell level'''
    if not matrix:
        strand_str = 'stranded' if stranded else 'unstranded'
        out_path = os.path.join(out_dir, f'site_{strand_str}_{mode}.tsv.gz')
        output_site_table(
            site_dir, 
            stranded, 
            mode, 
            out_path, 
            celltype_map,
        )
    else:
        output_cell_by_site_matrix(
            site_dir, 
            out_dir, 
            celltype_map,
        )
