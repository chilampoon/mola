import click
import os
import logging
from mola.read.process_alignment import *
from mola.read.annotate_read import annotate_reads
from mola.read.trim_read import trim_reads

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class Config(object):
    def __init__(self):
        self.verbose = False

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group('read')
@click.option('--min_read_len', show_default=True, default=0,
              help="Minimal read length")
@click.option('--min_mapq', show_default=True, default=20,
              help="Minimal mapping quality score")
@click.option('--secondary', is_flag=True, show_default=True, default=False,
              help="Include secondary and supplementary alignemnts")
@click.option('-o', '--out_dir', default=os.getcwd(), 
              help="Output directory, default is current working directory")
@pass_config
def mola_read(config, min_read_len, min_mapq, secondary, out_dir):
    '''
    Read/alignment processing
    '''
    os.makedirs(out_dir, exist_ok=True)
    # tags:
    # CB: corrected barcode; CR: uncorrected barocde
    # UB: corrected UMI;     UR: uncorrected UMI

    # set up config
    primary_only = True if not secondary else False
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    config.min_len = min_read_len
    config.min_mapq = min_mapq
    config.primary = primary_only
    config.out_dir = out_dir


@mola_read.command('demux')
@click.option('-b', '--bam', required=True, type=click.Path(exists=True),
              help="BAM file")
@click.option('--tag_demux', required=True, default='CB', show_default=True,
              help="Tag id that is used to split")
@pass_config
def demux(config, bam, tag_demux):
    '''Demultiplex a bam by tags'''
    demux_by_tag(bam, tag_demux, config.out_dir)


@mola_read.command('subset')
@click.option('-b', '--bam', required=True, type=click.Path(exists=True),
              help="BAM file")
@click.option('--read_id_file', default=None, type=click.Path(exists=True),
              help="File containing read ids")
@pass_config
def subset(config, bam, read_id_file):
    '''Extract alignments by tag or id'''
    subset_bam(bam, config.out_dir, read_id_file)


@mola_read.command('trim')
@click.option('-b', '--bam', required=True, type=click.Path(exists=True),
              help="BAM file")
@click.option('--max_edit_dist', default=2, show_default=True,
              help="Maximal edit distance for sequence matching")
@click.option('--lib', default='10x_3prime_v3', show_default=True,
              help="Sequencing library kit")
@click.option('--short', is_flag=True, show_default=True, default=False,
              help="Short reads")
@pass_config
def trim(config, bam, max_edit_dist, lib, short):
    '''Trim primer, adapter, polyA/T, tso, etc'''
    out_dir = config.out_dir
    trim_reads(bam, out_dir, lib, max_edit_dist, short)


@mola_read.command('annotate')
@click.option('-b', '--bam', required=True, type=click.Path(exists=True),
              help="BAM file")
@click.option('--bulk', is_flag=True, show_default=True, default=False,
              help="If specify, process bulk RNA-seq")
@click.option('--paired_end', is_flag=True, show_default=True, default=False,
              help="If specify, process paired-end bulk RNA-seq")
@click.option('--repeat_bed', required=True, type=click.Path(exists=True),
              help="bed file for repeat annotation, e.g. repeatmasker")
@click.option('--gff3', required=True, type=click.Path(exists=True),
              help="gff3/gtf file for gene annotation")
@click.option('--read_assignments', default=None, type=click.Path(exists=True),
              help="Long read assignment output from isoquant; reads are assigned to genes confidently")
@click.option('--min_overlap', type=float, default=0.3, show_default=True,
              help="Minimum overlap threshold for the shorter feature (exon or repeat) during intersection")
@click.option('--min_exon_on_read', type=float, default=0.85, show_default=True,
              help="Minimum exon ratio in the read to call it a exonic read, for bulk short reads")
@click.option('--min_repeat_on_read', type=float, default=0.3, show_default=True,
              help="Minimum repeat ratio in the read to call it a repeat read")
@click.option('--min_intron_on_read', type=float, default=0.8, show_default=True,
              help="Minimum intron ratio in the read to call it an intron read")
@click.option('--min_intron_unspliced', type=float, default=0.1, show_default=True,
              help="Minimum intron ratio in any read exon to call it unspliced")
@click.option('--alu_merge_dist', type=str, default='500', show_default=True,
              help="Distance for merging alu elements")
@click.option('--not_startswith_chr', is_flag=True, show_default=True, default=False,
              help="If specify, all chromosome ids must not start with 'chr'")
@click.option('-t', '--num_threads', type=int, default=None, show_default=True,
              help="Number of threads for parallel processing")
@click.option('--tmp_dir', type=click.Path(exists=True), default=None,
              help="Previous tmp dir containing all annotation and alignment bed")
@pass_config
def process_and_annotate(config, bam, bulk, paired_end, repeat_bed, gff3, read_assignments,
        min_overlap, min_exon_on_read, min_repeat_on_read, min_intron_on_read, min_intron_unspliced,
        alu_merge_dist, not_startswith_chr, num_threads, tmp_dir, return_tmp_dir=False
    ):
    '''Process alignments to Reads and annotate'''
    startswith_chr = not not_startswith_chr
    tmp_dir = annotate_reads(
        bam,
        bulk,
        paired_end,
        config.primary,
        config.min_len,
        config.min_mapq,
        read_assignments,
        gff3,
        repeat_bed,
        alu_merge_dist,
        min_overlap,
        min_exon_on_read, 
        min_repeat_on_read,
        min_intron_on_read,
        min_intron_unspliced,
        num_threads,
        startswith_chr,
        tmp_dir,
        config.out_dir
    )

    if return_tmp_dir:
        return tmp_dir
