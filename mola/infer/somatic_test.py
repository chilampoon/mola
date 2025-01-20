from mola.parse.file_utils import *
from scipy.stats import chi2_contingency, fisher_exact, false_discovery_control
import matplotlib.pyplot as plt
from mola.infer.snv_model_pyro import *
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

futils = FileUtils()
MIN_TEST_COUNT = 10

"""
Test for somatic mutations post phasing (Aug 2024)

Input:
  - Reads
  - Sites
  - haplo.tsv
  - posterior_probs.tsv.gz

Output:
  - Table of soma test results combined with betabinom out
  - Actual haplotype count table
  - Plots
"""


def simple_stats(cnt_df):
    # minor haplotype
    minor_hap = cnt_df.sum(axis=0).idxmin()
    minor_hap_sum = cnt_df[minor_hap].sum()
    minor_hap_maf = cnt_df[minor_hap].min() / minor_hap_sum
    
    # minor allele
    minor_allele = cnt_df.sum(axis=1).idxmin()
    maf = cnt_df.loc[minor_allele,:].sum() / cnt_df.sum().sum()
    
    # major allele
    major_allele = cnt_df.sum(axis=1).idxmax()
    major_allele_sum = cnt_df.loc[major_allele,:].sum()
    major_allele_maf = cnt_df.loc[major_allele,:].min() / major_allele_sum
    
    return maf, major_allele_maf, major_allele_sum, minor_hap_maf, minor_hap_sum

def check_table_cnts(cnt_df):
    '''
    if it's germline variant, counts would be like
            h1  |  h2
        a1  100 |  0
        a2  0   |  100
    so pass them
    '''
    coverage_thres = 10
    minor_thres = 2
    # compute column sums
    col_sums = cnt_df.sum(axis=0)
    # check if any column sum is less than 10
    enough_coverage = False if (col_sums < coverage_thres).any() else True
    
    # check if the diagonal elements of the df are all zeros
    is_germline = ((cnt_df.iloc[0, 0] != 0 and cnt_df.iloc[1, 1] != 0) and \
                    (cnt_df.iloc[0, 1] < minor_thres and cnt_df.iloc[1, 0] < minor_thres)) or \
                ((cnt_df.iloc[0, 0] < minor_thres and cnt_df.iloc[1, 1] < minor_thres) and \
                    (cnt_df.iloc[0, 1] != 0 and cnt_df.iloc[1, 0] != 0))
    test_it = True if enough_coverage and not is_germline else False
    return test_it

def classify_mut(hap_mut_rates, error_rate):
    hap_mut_rates = hap_mut_rates - error_rate
    delta = hap_mut_rates.max() - hap_mut_rates.min()
    if hap_mut_rates.min() < 0.05 and delta > 0.1:
        return 'somatic'
    else:
        return 'error'

def process_chromosome(chrom, reads, sites, haplo, posterior, 
                    n_haplos, prop_mut, seq_error_rate, learning_rate, 
                    num_steps, out_dir):
    # get all non-error sites
    logging.info(f'chromosome {chrom}...')
    
    hap_cnt = []
    soma_test_res = []
    stats = []
    
    # sites to avoid to test
    phasable_sites = haplo[haplo['locus_flag'] == 'MP'][['chr','pos']]
    germline_sites = [f'{c}:{pp}' for c, p in zip(phasable_sites['chr'], phasable_sites['pos']) for pp in p.split(',')]
    # haps to avoid
    hap_avoid = haplo[haplo['locus_flag'] == 'S'][['pos', 'locus']]
    hap_avoid_real = [p for p in hap_avoid['pos'].values if posterior.loc[posterior['pos']==str(p), 'snp_type'].values[0] != 'common']
    hap_avoid = hap_avoid[hap_avoid['pos'].isin(hap_avoid_real)]
    hap_avoid_loci = hap_avoid['locus'].to_list()
    logging.info(f'{len(hap_avoid_loci)} seemed-not-good loci to skip...')
    
    for idx, row in posterior.iterrows():
        if row['snp_type'] == 'common':
            continue
        
        s = int(row['pos'])
        site_pos = f'{chrom}:{s}'
        if site_pos in germline_sites:
            continue
        
        site_obj = sites[(chrom, s)]
        a1, a2 = site_obj.mismatch.split('>')
        
        # get alleles counts on each haplotype
        site_hap_cnts = defaultdict(lambda: defaultdict(lambda:{a1:0, a2:0}))
        reads_with_site = defaultdict(list)
        for r in site_obj.reads:
            read_id, strand, base = r.split('|')
            read_obj = reads.get(read_id)
            if not read_obj.hap:
                continue
            
            for locus, hap_id in read_obj.hap.items():
                loc_hap = f'{locus}_{hap_id}'
            ## old version ##
            # for loc_hap in read_obj.hap:
            #     locus = loc_hap.replace('_1', '').replace('_2', '').replace('_.', '')
                if base in [a1, a2]:
                    site_hap_cnts[locus][loc_hap][base] += 1
                
                base_num = 0 if base == a1 else 1 if base == a2 else 2
                hap_num = 0 if hap_id == '1' else 1 if hap_id == '2' else 2
                reads_with_site[locus].append([base_num, hap_num])
                
        if not site_hap_cnts:
            continue
        
        # test for somatic mutations
        for locus, haplo_cnts in site_hap_cnts.items():
            if locus.replace('loc', '') in hap_avoid_loci:
                print(f'skipping {locus}...')
                continue
            
            cnt_df = pd.DataFrame(haplo_cnts)
            cnt_total = sum(cnt_df.sum())
            if cnt_df.shape[1] > 2:
                cnt_df = cnt_df.iloc[:, :2]
            elif cnt_df.shape[1] < 2:
                continue
            
            if not check_table_cnts(cnt_df):
                continue
            
            # run somatic test
            data = torch.tensor(reads_with_site[locus])
            hap_mut_rates, hap_ratio, error_rate = somatic_test_svi(
                data, n_haplos=n_haplos,
                model=snv_model, 
                prop_mut=prop_mut, error_rate=seq_error_rate,
                random_seed=21, lr=learning_rate, n_steps=num_steps
            )
            mut_cat = classify_mut(hap_mut_rates, error_rate)

                
            # collect simple stats
            maf, major_allele_maf, major_allele_sum, minor_hap_maf, minor_hap_sum = simple_stats(cnt_df)
            stats.append([chrom, row['snv'], locus, row['gene'], maf, 
                        major_allele_maf, major_allele_sum, minor_hap_maf, minor_hap_sum])
            # collect hap cnt tables
            cnt_df.columns = ['h1', 'h2']
            cnt_df['snv'] = row['snv']
            cnt_df['gene'] = row['gene']
            cnt_df['locus'] = locus
            hap_cnt.append(cnt_df)
            if locus.replace('loc', '') not in haplo['locus'].values:
                locus_flag = '.'
            else:
                locus_flag = haplo.loc[haplo['locus'] == locus.replace('loc', ''), 'locus_flag'].values[0]
            
            res_row = [row['snv'], row['gene'], cnt_total, locus, locus_flag, 
                        row['region'], row['snp_type'], row['category_prob'], 
                        ','.join(['%.3f' % r for r in hap_mut_rates]),
                        ','.join(['%.3f' % r for r in hap_ratio]), error_rate, mut_cat]
            #print(res_row)
            soma_test_res.append(res_row)
            
    hap_cnt_out = os.path.join(out_dir, f'{chrom}_hap_cnt.tsv')
    if hap_cnt:
        hap_cnt = pd.concat(hap_cnt, axis=0)
        hap_cnt.index.name = 'allele'
        hap_cnt.to_csv(hap_cnt_out, sep='\t', index=True, header=False)
    soma_test_res = pd.DataFrame(soma_test_res)
    return soma_test_res, pd.DataFrame(stats)

def soma_test(reads_dir, sites_dir, haplo_path, posterior_path, n_haplos, 
            prop_mut, seq_error_rate, learning_rate, num_steps, out_dir):
    logging.info(f'testing somatic mutations...')
    # load inputs
    haplo = pd.read_csv(haplo_path, dtype={1: str}, sep='\t')
    posterior = pd.read_csv(posterior_path, sep="\t").drop_duplicates()
    
    hap_cnt_out = os.path.join(out_dir, 'hap_cnt.tsv')
    soma_test_out = os.path.join(out_dir, 'soma_test.tsv')
    tmp_dir = futils.make_tmp_dir(out_dir)
    
    #sites_test = posterior[posterior['category'] != 'error']
    posterior[['chrom', 'pos']] = [m.split('|')[0].split(':') for m in posterior['snv']]
    sites_test = posterior[posterior['chrom'].isin(haplo['chr'].values)]
    
    soma_test_res = []
    stats_res = []
    sites_grouped = sites_test.groupby('chrom')
    # TODO - parallelize
    for chrom, posteriors_chrom in sites_grouped:
        if chrom != 'chr17':
            continue
        reads_obj_path = os.path.join(reads_dir, f'{chrom}_reads.pkl.gz')
        reads = futils.load_gz_pickle(reads_obj_path)
        sites_obj_path = os.path.join(sites_dir, f'{chrom}_sites.pkl.gz')
        sites = futils.load_gz_pickle(sites_obj_path)
        
        haplo_chrom = haplo[haplo['chr'] == chrom]
        res, stats = process_chromosome(chrom, reads, sites, haplo_chrom, posteriors_chrom, 
                    n_haplos, prop_mut, seq_error_rate, learning_rate, num_steps, out_dir)
        soma_test_res.append(res)
        stats_res.append(stats)
        
    hap_cnt_out = os.path.join(out_dir, 'hap_cnt.tsv')
    hap_cnt_header = ['allele', 'hap1_cnt','hap2_cnt','snv', 'gene', 'locus']
    with open(hap_cnt_out, 'w') as out:
        out.write(futils.list2line(hap_cnt_header))
    
    hap_cnt_chrom_files = sorted(
        glob.glob(f'{tmp_dir}/*hap_cnt.tsv'), 
        key=futils.sort_chrom_files
    )
    futils.concat_files(hap_cnt_chrom_files, hap_cnt_out)
    
    stats_res = pd.concat(stats_res)
    stats_res.columns = ['chrom', 'snv', 'locus', 'gene','maf', 'major_allele_maf', 'major_allele_sum','minor_hap_maf', 'minor_hap_sum']
    stats_res.to_csv(f"{out_dir}/stats.tsv", sep='\t', index=False)
    
    soma_test_res = pd.concat(soma_test_res).reset_index(drop=True)
    soma_test_res.columns = ['snv', 'gene', 'coverage', 'locus', 'locus_flag', 'region', 
                            'snp_type', 'category_prob', 'hap_mut_rates', 'hap_ratio', 
                            'error_rate', 'category_mut']
    soma_test_res.fillna('.').to_csv(soma_test_out, sep='\t', index=False)

    shutil.rmtree(tmp_dir)
