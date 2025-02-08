import click
import os
import logging
from mola.infer.phaser import *
from mola.infer.calc_post_prob import ProbCalculator
from mola.infer.somatic_test import soma_test

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


@click.group('infer')
def mola_infer():
    '''
    Phasing long cDNA/mRNA reads with a Bayesian graphical model
    '''
    pass

common_options = [
    click.option('-s', '--sites_stranded', required=True, type=click.Path(exists=True),
                help="Stranded sites file, output from longr-mut map"),
]
def common_options_decorator(func):
    for option in reversed(common_options):
        func = option(func)
    return func


@mola_infer.command('posteriors')
@common_options_decorator
@click.option('--tech', show_default=True, default='pacbio',
              help="pacbio, ont, or illumina")
@click.option('-u', '--sites_unstranded', required=False, type=click.Path(exists=True),
              help="Unstranded sites file, output from longr-mut map")
@click.option('--num_nonsnp_comps', type=int, default=2, show_default=True,
              help="Number of non-SNP components in one mixture distribution, i.e. one type of mismatch")
@click.option('--edit_pair', show_default=True, default='A>G,G>A',
              help="Substitution pair of editing, e.g. A>G,G>A, separated by comma")
@click.option('--min_nonerr_weight', type=float, default=0, show_default=True,
              help="Minimum non-error weight in the mixture")
@click.option('--min_coverage', type=int, default=10, show_default=True,
              help="Minimum coverage of mismatches")
@click.option('--min_minor_cnt', type=int, default=2, show_default=True,
              help="Minimum minor allele counts")
@click.option('--min_minor_af', type=float, default=0.01, show_default=True,
              help="Minimum minor allele frequency of mismatches")
@click.option('--learning_rate', type=float, default=0.01, show_default=True,
              help="Learning rate in variational inference")
@click.option('--num_steps', type=int, default=3200, show_default=True,
              help="Number of steps in variational inference")
@click.option('--plot_xlim', type=str, default='0,0.5', show_default=True,
              help="X-axis limit for plotting, separated by comma")
@click.option('--plot_ylim', type=str, default='0,1000', show_default=True,
              help="Y-axis limit for plotting, separated by comma")
@click.option('-o', '--out_dir', default=os.getcwd(), 
              help="Output directory, default is current working directory")
def calculate_posteriors(tech, sites_stranded, sites_unstranded, num_nonsnp_comps, edit_pair, min_nonerr_weight,
                        min_coverage, min_minor_cnt, min_minor_af, learning_rate, num_steps, 
                        plot_xlim, plot_ylim, out_dir):
    '''Calculate posterior probabilities of being germline variant, error, or editing
    '''
    os.makedirs(out_dir, exist_ok = True)
    plot_xlim = [float(x) for x in plot_xlim.split(',')]
    plot_ylim = [float(x) for x in plot_ylim.split(',')]
    cal = ProbCalculator(tech, learning_rate, num_steps, min_nonerr_weight, num_nonsnp_comps, 
                        edit_pair, plot_xlim, plot_ylim, out_dir)
    cal.get_posterior_probs(sites_stranded, sites_unstranded, min_coverage, 
                            min_minor_cnt, min_minor_af)


@mola_infer.command('phayes')
@click.option('-d', '--obj_dir', required=True, type=click.Path(exists=True),
              help="Object directory, output from read annotation and mutation mapping")
@click.option('--posterior_probs', type=click.Path(exists=True), required=True, 
              help="Path to site dataframe with posterior probabilities")
@click.option('-k', '--num_haplotypes', type=int, default=2, show_default=True, 
              help="Number of haplotypes")
@click.option('--stranded', is_flag=True, show_default=True, default=False,
              help="Reads are stranded, default is unstranded")
@click.option('--edit_pair', type=str, default='A>G,G>A', required=True, 
              help="RNA editing pair, e.g. A>G")
@click.option('--num_steps', type=int, default=3000, show_default=True,
              help="Number of steps in SVI")
@click.option('--learning_rate', type=float, default=0.01, show_default=True,
              help="Learning rate in SVI")
@click.option('--max_phasing_times', type=int, default=5, show_default=True,
              help="Maximum number of phasing for multisnp loci")
@click.option('-t', '--num_threads', type=int, default=None, show_default=True,
              help="Number of threads for parallel processing")
@click.option('-o', '--out_dir', default=os.getcwd(), 
              help="Output directory, default is current working directory")
def phayes(**args):
    '''Phasing + Bayesian = Phayesing!'''
    os.makedirs(args.get('out_dir'), exist_ok = True)

    phase(
        args.get('num_haplotypes'),
        args.get('edit_pair'), 
        args.get('learning_rate'), 
        args.get('num_steps'), 
        args.get('max_phasing_times'), 
        args.get('stranded'),
        args.get('posterior_probs'), 
        args.get('obj_dir'), 
        args.get('num_threads'), 
        args.get('out_dir')
    )


@mola_infer.command('soma')
@click.option('-d', '--obj_dir', required=True, type=click.Path(exists=True),
              help="Object directory, output from read annotate")
@click.option('--haplo', required=True, type=click.Path(exists=True),
              help="Path to haplo.tsv, output from infer phayes")
@click.option('--post_probs', required=True, type=click.Path(exists=True),
              help="Path to site dataframe with posterior probabilities, output from infer posteriors")
@click.option('--bb_params', required=True, type=click.Path(exists=True),
              help='Path to betabinom_params.json, output from infer posteriors')
@click.option('--mut_prop', required=True, type=float, default=0.1, show_default=True,
              help="Proportion of somatic mutations, ranging from 0 to 1")
@click.option('--n_haplos', required=True, type=int, default=2, show_default=True,
              help="Number of haplotypes")
@click.option('--learning_rate', type=float, default=0.05, show_default=True,
              help="Learning rate in SVI")
@click.option('--num_steps', type=int, default=300, show_default=True,
              help="Number of steps in variational inference")
@click.option('-o', '--out_dir', default=os.getcwd(), 
              help="Output directory, default is current working directory")
def run_somatic_test(obj_dir, haplo, post_probs, bb_params, mut_prop, n_haplos, 
                    learning_rate, num_steps, out_dir):
    '''Test for somatic mutations'''
    out_dir = os.path.join(out_dir, 'somatic_test')
    os.makedirs(out_dir, exist_ok=True)
    reads_dir = os.path.join(obj_dir, 'reads')
    sites_dir = os.path.join(obj_dir, 'sites')
    soma_test(reads_dir, sites_dir, haplo, post_probs, bb_params, mut_prop, 
            n_haplos, learning_rate, num_steps, out_dir)
