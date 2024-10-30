import click
import os
import logging
from mola.read.exprs_matrix import *

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@click.group('parse')
def mola_parse():
    '''
    Parse files
    '''
    pass


@mola_parse.command('count')
@click.option('-d', '--obj_dir', required=True, type=click.Path(exists=True),
              help="Object directory, output from read annotation")
@click.option('--assay', required=True, default='sc',
              type=click.Choice(['sc', 'bulk'], case_sensitive=False))
@click.option('--paired_end', is_flag=True, show_default=True, default=False,
              help="Paired end reads (bulk)")
@click.option('--splice', is_flag=True, show_default=True, default=False,
              help="Generate spliced and unspliced count matrices, force to be 'gene' mode. Only sc for now.")
@click.option('--mode', show_default=True, default='gene',
              help="Count reads from 'gene', 'Alu', 'intron', 'oth', 'intergenic', 'an' with operations of + or (+)")
@click.option('--concat_prefixes', show_default=True, default=None,
              help="Concatenate prefixes for features, separated by comma, e.g. gene,nc")
@click.option('--gene_id', is_flag=True, show_default=True, default=False,
              help="Use gene ids instead of gene names in the matrix")
@click.option('--barcode_list', required=False, type=click.Path(exists=True),
              help="A list of cell barcodes to include in the matrix")
@click.option('-o', '--out_dir', default=os.getcwd(), 
              help="Output directory, default is current working directory")
@click.option('-t', '--num_threads', type=int, default=None, show_default=True,
              help="Number of threads for parallel processing")

def get_cnt_matrix(obj_dir, assay, paired_end, splice, mode, concat_prefixes, 
                gene_id, barcode_list, out_dir, num_threads):
    '''Get feature expression matrix'''
    os.makedirs(out_dir, exist_ok=True)
    
    if assay == 'sc':
        get_cell_by_gene_matrix(
            obj_dir=obj_dir,
            splice=splice,
            mode=mode,
            concat_prefixes=concat_prefixes,
            out_dir=out_dir, 
            gene_id=gene_id,
            barcode_list=barcode_list,
            num_threads=num_threads
        )
    else:
        get_gene_matrix(
            obj_dir=obj_dir,
            paired_end=paired_end,
            mode=mode,
            concat_prefixes=concat_prefixes,
            out_dir=out_dir,
            gene_id=gene_id,
            num_threads=num_threads
        )
