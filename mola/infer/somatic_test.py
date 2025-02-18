from mola.parse.file_utils import *
from mola.infer.snv_model_pyro import *
import json
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

futils = FileUtils()
MIN_TEST_COUNT = 10
EDITS = ['A>G', 'G>A', 'T>C', 'C>T']

"""
Test for somatic mutations post phasing (Aug 2024, updated in 2025)

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

def process_chromosome(chrom, reads, sites, haplo, posterior, mut_prop, bb_params,
                    n_haplos, learning_rate, num_steps, out_dir):
    # get all non-error sites
    logging.info(f'chromosome {chrom}...')
    hap_cnt = []
    soma_test_res = []
    stats = []
    
    # sites to avoid to test
    phasable_sites = haplo[haplo['locus_flag'] == 'MP'][['chr','pos']]
    germline_sites = [f'{c}:{pp}' for c, p in zip(phasable_sites['chr'], phasable_sites['pos']) for pp in p.split(',')]
    # haps to avoid (not common snp single locus)
    hap_avoid = haplo[haplo['locus_flag'] == 'S'][['pos', 'locus']]
    hap_avoid_real = [p for p in hap_avoid['pos'].values if posterior.loc[posterior['pos']==str(p), 'snp_type'].values[0] != 'common' \
                    or posterior.loc[posterior['pos'] == str(p), 'minor_af'].astype(float).values[0] < 0.35]
    hap_avoid = hap_avoid[hap_avoid['pos'].isin(hap_avoid_real)]
    hap_avoid_loci = hap_avoid['locus'].to_list()
    logging.info(f'{len(hap_avoid_loci)} seemed-not-good loci to skip...')
    
    germline_tested = [] 
    for idx, row in posterior.iterrows():
        if filter_post_row(row):
            continue
        
        s = int(row['pos'])
        site_pos = f'{chrom}:{s}'
        if site_pos in germline_sites:
            continue
        
        site_obj = sites[(chrom, s)]
        a1, a2 = site_obj.mismatch.split('>')
        # process prior info
        error_params, edit_prior = bb_params['error'], bb_params['edit']
        error_prior = error_params[site_obj.mismatch]
        event_probs = [(1-mut_prop)/2, mut_prop, (1-mut_prop)/2]
        
        # get alleles counts on each haplotype
        site_hap_cnts = defaultdict(lambda: defaultdict(lambda:{a1:0, a2:0}))
        reads_with_site = defaultdict(list)
        for r in site_obj.reads:
            read_id, strand, base = r.split('|')
            read_obj = reads.get(read_id)
            if not read_obj.hap:
                continue
            
            for locus, hap_id in read_obj.hap.items():
                if locus.replace('loc', '') not in haplo['locus'].values:
                    continue
                loc_hap = f'{locus}_{hap_id}'
                
                if base in [a1, a2]:
                    site_hap_cnts[locus][loc_hap][base] += 1
                    # NOTE - not modeling other bases for now
                    base_num = 0 if base == a1 else 1 if base == a2 else 2
                    hap_num = 0 if hap_id == '1' else 1 if hap_id == '2' else 2
                    reads_with_site[locus].append([base_num, hap_num])
                
        if not site_hap_cnts:
            continue
        
        # test for somatic mutations
        for locus, haplo_cnts in site_hap_cnts.items():
            if locus.replace('loc', '') in hap_avoid_loci or locus.replace('loc', '') not in haplo['locus'].values:
                continue
            
            cnt_df = pd.DataFrame(haplo_cnts)
            cnt_df = cnt_df.reindex(sorted(cnt_df.columns), axis=1)
            cnt_total = sum(cnt_df.sum())
            if cnt_df.shape[1] > 2:
                cnt_df = cnt_df.iloc[:, :2]
            elif cnt_df.shape[1] < 2:
                continue
            
            if is_germline_tab(cnt_df):
                # save SNV to clean up record if any
                germline_tested.append(row['snv'])
                break
            if not has_good_coverage(cnt_df):
                continue
            
            event_probs = update_probs(event_probs, cnt_df)
            
            # run somatic test
            data = torch.tensor(reads_with_site[locus])
            params = somatic_test_svi(
                data, model=snv_diploid_model, 
                event_probs=event_probs,
                error_prior=error_prior,
                edit_prior=edit_prior,
                random_seed=21, lr=learning_rate, n_steps=num_steps
            )
            prediction, soma_prob, τ_e, τ_h, τ_b, μ_mut, μ_edit, ε = parse_svi_out(params)
            
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
            
            locus_flag = haplo.loc[haplo['locus'] == locus.replace('loc', ''), 'locus_flag'].values[0]
            res_row = [row['snv'], row['gene'], cnt_total, locus, locus_flag, 
                        row['region'], row['snp_type'], row['category'], 
                        prediction, soma_prob, τ_e, τ_h, τ_b, μ_mut, μ_edit, ε]
            soma_test_res.append(res_row)
    
    hap_cnt_out = os.path.join(out_dir, f'{chrom}_hap_cnt.tsv')
    if hap_cnt:
        hap_cnt = pd.concat(hap_cnt, axis=0)
        hap_cnt.index.name = 'allele'
        hap_cnt = hap_cnt[~hap_cnt['snv'].isin(germline_tested)] # kick out germlines
        hap_cnt.to_csv(hap_cnt_out, sep='\t', index=True, header=False)
    soma_test_res = pd.DataFrame(soma_test_res)
    if not soma_test_res.empty:
        soma_test_res = soma_test_res[~soma_test_res.iloc[:, 0].isin(germline_tested)] # kick out germlines
    return soma_test_res, pd.DataFrame(stats)

def update_probs(event_probs, cnt_df):
    '''
    if maf on both haplotypes are high, its more like an edit
    reduce the mutation probability
    (or it's heterozygous mutation??)
    '''
    maf_thres = 0.1
    col_sums = cnt_df.sum(axis=0)
    row_sums = cnt_df.sum(axis=1)
    row_minor = cnt_df.loc[row_sums.idxmin()]
    row_minor_pert = row_minor / col_sums
    if (row_minor_pert >= maf_thres).all():
        event_probs[1] /= 10
    return event_probs

def digest_betabinom_params(bb_params):
    '''
    beta binomial params estimated from the mixture model.
    serve as error prior for the SNV model here
    '''
    with open(bb_params) as f:
        params = json.load(f)
    
    error_params = {}
    for category, values in params['error_params'].items():
        weights = values['weights']
        alphas = values['alphas']
        betas = values['betas']
        # pick those with the highest weight
        max_index = weights.index(max(weights))
        error_params[category] = (alphas[max_index], betas[max_index])
    
    # for edit, pick the one with the highest mean regardless of categories
    edit_params = {}
    max_mean = 0
    for category, values in params['edit_params'].items():
        weights = values['weights']
        alphas = values['alphas']
        betas = values['betas']
        
        candidates = [
            (a, b, a / (a + b))  # store alpha, beta, and pre-computed mean
            for w, a, b in zip(weights, alphas, betas)
            if w >= 0.01
        ]
        if candidates:
            a, b, highest_mean = max(candidates, key=lambda x: x[2])
        
        if highest_mean > max_mean:
            max_mean = highest_mean
            best_alpha, best_beta = a, b
    
    if max_mean == 0:
        best_alpha, best_beta = 2, 10
    edit_params = (best_alpha, best_beta)
    return {'error': error_params, 'edit':edit_params}

def filter_post_row(posterior_row):
    '''
    filter out common snp and mismatch with > 2 alleles 
    '''
    if posterior_row['snp_type'] == 'common':
        return True # T -> throw away
    bases = posterior_row[['A', 'C', 'G', 'T', 'N']].to_numpy(dtype=int)
    base_pert = bases / bases.sum()
    base_pert = np.sort(base_pert[base_pert > 0])[::-1]
    # if the third highest allele frequency is greater than 0.08, remove
    if len(base_pert) > 2 and base_pert[2] >= 0.08:
        return True

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

def is_germline_tab(cnt_df):
    '''
    if it's germline variant, counts would be like
            h1  |  h2
        a1  100 |  0
        a2  0   |  100
    so pass them
    '''
    coverage_thres_minor = 5
    col_sums = cnt_df.sum(axis=0)
    minor_thres = 0.03 # at most 3% minor allele frequency
    
    # check if the diagonal elements of the df are all zeros
    col_mins = cnt_df.min(axis=0)
    min_indices = cnt_df.idxmin(axis=0)
    min_arr = np.array([max(i, coverage_thres_minor) for i in col_sums * minor_thres])
    is_germline = (col_mins < min_arr).all() and min_indices.nunique() > 1
    return is_germline

def has_good_coverage(cnt_df):
    coverage_thres_sum = 30
    coverage_thres_minor = 5

    # compute column sums
    col_sums = cnt_df.sum(axis=0)
    row_sums = cnt_df.sum(axis=1)
    row_minor = cnt_df.loc[row_sums.idxmin()]
    good_coverage = False if (col_sums < coverage_thres_sum).any() or (row_minor < coverage_thres_minor).all() else True
    return good_coverage

def parse_svi_out(params):
    preds = ['error', 'somatic', 'edit']
    τ_e, τ_h, τ_b, ε, μ_edit_h0, μ_edit_h1, μ_mut = params
    
    prediction = preds[np.argmax(τ_e).item()]
    soma_prob = τ_e[1].item()
    τ_e = ':'.join(map(lambda x:"%.3f" % x, τ_e))
    τ_h = ':'.join(map(lambda x:"%.3f" % x, τ_h))
    τ_b = ':'.join(map(lambda x:"%.3f" % x, τ_b))
    ε = ε.item()
    μ_edit_h0 = μ_edit_h0.item()
    μ_edit_h1 = μ_edit_h1.item()
    μ_edit = f'{μ_edit_h0:.3f}:{μ_edit_h1:.3f}'
    μ_mut = μ_mut.item()
    return prediction, soma_prob, τ_e, τ_h, τ_b, μ_mut, μ_edit, ε

def filter_somatic(soma_test_res):
    '''
    for each SNV, all of its prediction(s) need to be somatic to call it a somatic mutation
    otherwise removes the somatic prediction row
    '''
    snv_all_somatic = soma_test_res.groupby('snv')['prediction'].transform(
        lambda x: (x == 'somatic').all()
    )
    filtered_df = soma_test_res[(snv_all_somatic) | (soma_test_res['prediction'] != 'somatic')]
    return filtered_df.reset_index(drop=True)
    
def soma_test(reads_dir, sites_dir, haplo_path, posterior_path, bb_params_path,
            mut_prop, n_haplos, learning_rate, num_steps, out_dir):
    logging.info(f'testing somatic mutations...')
    # load inputs
    haplo = pd.read_csv(haplo_path, dtype={1: str}, sep='\t')
    posterior = pd.read_csv(posterior_path, sep="\t").drop_duplicates()
    
    hap_cnt_out = os.path.join(out_dir, 'hap_cnt.tsv')
    soma_test_out = os.path.join(out_dir, 'soma_test.tsv')
    tmp_dir = futils.make_tmp_dir(out_dir)
    
    bb_params = digest_betabinom_params(bb_params_path)
    
    #sites_test = posterior[posterior['category'] != 'error']
    posterior[['chrom', 'pos']] = [m.split('|')[0].split(':') for m in posterior['snv']]
    sites_test = posterior[posterior['chrom'].isin(haplo['chr'].values)]
    
    soma_test_res = []
    stats_res = []
    sites_grouped = sites_test.groupby('chrom')
    
    for chrom, posteriors_chrom in sites_grouped:
        reads_obj_path = os.path.join(reads_dir, f'{chrom}_reads.pkl.gz')
        reads = futils.load_gz_pickle(reads_obj_path)
        sites_obj_path = os.path.join(sites_dir, f'{chrom}_sites.pkl.gz')
        sites = futils.load_gz_pickle(sites_obj_path)
        
        haplo_chrom = haplo[haplo['chr'] == chrom]
        res, stats = process_chromosome(chrom, reads, sites, haplo_chrom, posteriors_chrom, 
                    mut_prop, bb_params, n_haplos, learning_rate, num_steps, tmp_dir)
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
    soma_test_res.columns = ['snv', 'gene', 'coverage', 'locus', 'locus_flag', 'region', 'snp_type', 
                            'category', 'prediction', 'soma_prob', 'event_probs', 'hap_preference', 
                            'base_preference', 'mut_frq', 'edit_frqs', 'error_frq']
    soma_test_res = filter_somatic(soma_test_res)
    soma_test_res.fillna('.').to_csv(soma_test_out, sep='\t', index=False)

    shutil.rmtree(tmp_dir)
