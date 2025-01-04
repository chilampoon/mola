import pandas as pd
pd.options.mode.chained_assignment = None
from igraph import Graph
import os, glob, shutil
import logging
from collections import defaultdict, Counter, deque
from mola.mutation.mismatch_site import REV_COMP_BASES
from mola.infer.phasing_model_pyro import *
from mola.parse.file_utils import FileUtils
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

futils = FileUtils()

def phase(num_haplotypes, edit_pair, learning_rate, 
        num_steps, max_phasing_times, stranded,
        posterior_probs, obj_dir, num_threads, out_dir
    ):
    '''
    Phase (relatively) germline snps
    '''
    out_dir = os.path.join(out_dir, 'phasing_out')
    os.makedirs(out_dir, exist_ok=True)

    # get germline snps from posterior probabilities
    germline_df = get_germline_df(posterior_probs)
    chrom_list = germline_df['chrom'].unique().tolist()
    chrom_list = futils.sort_chroms(chrom_list)
    
    tmp_dir = futils.make_tmp_dir(out_dir=out_dir, suffix='_phasing')
    _ = futils.process_chrom_in_threads(
        process_chromosome,
        chrom_list,
        num_threads,
        germline_df=germline_df, 
        num_haplotypes=num_haplotypes,
        edit_pair=edit_pair,
        learning_rate=learning_rate,
        num_steps=num_steps,
        max_phasing_times=max_phasing_times,
        stranded=stranded,
        obj_dir=obj_dir, 
        out_dir=tmp_dir
    )
    
    hap_out = os.path.join(out_dir, 'haplo.tsv')
    hap_header = ['chr','locus','gene','pos'] + [f'hap_{h+1}' for h in range(num_haplotypes)] + ['hap_ratio','hap_cnts','locus_flag']
    with open(hap_out, 'w') as out:
        out.write('\t'.join(hap_header) + '\n')
    
    hap_chrom_files = sorted(
        glob.glob(f'{tmp_dir}/*.haplo.tsv'), 
        key=futils.sort_chrom_files
    )
    futils.concat_files(hap_chrom_files, hap_out)

    log_chrom_files = sorted(
        glob.glob(f'{tmp_dir}/*.phasing.log'), 
        key=futils.sort_chrom_files
    )
    futils.concat_files(log_chrom_files, os.path.join(out_dir, 'phasing.log'))

    df_chrom_files = sorted(
        glob.glob(f'{tmp_dir}/*.germlines.tsv.gz'), 
        key=futils.sort_chrom_files
    )
    futils.concat_files(df_chrom_files, os.path.join(out_dir, 'germlines.tsv.gz'))

    shutil.rmtree(tmp_dir)

def get_germline_df(posterior_probs):
    post_probs_df = pd.read_csv(posterior_probs, sep="\t").drop_duplicates()
    if 'category' in post_probs_df.columns:
        filtered_df = post_probs_df[post_probs_df['category'] == 'germline']
    else:
        prob_cols = [col for col in post_probs_df.columns if col.endswith('_post')]
        post_df = post_probs_df[prob_cols]
        # get those with max posterior prob for the germline one
        max_post = post_df.max(axis=1)
        filtered_df = post_probs_df[post_df['germline_post'] == max_post]

    # if sites fall into intergenic regions, they are not considered to get phased now.
    filtered_df[['chrom', 'pos']] = [snp.split('|')[0].split(':') for snp in filtered_df['snv']]
    filtered_df['pos'] = filtered_df['pos'].astype(int)
    return filtered_df

def process_chromosome(chrom, germline_df, num_haplotypes, edit_pair, learning_rate, 
                        num_steps, max_phasing_times, stranded, obj_dir, out_dir):
    logging.info(f'phasing snps on chromosome {chrom}...')
    germline_chrom = germline_df[germline_df['chrom']==chrom]

    # initiate a phaser for each chromosome
    phaser = Phaser(
        chrom,
        num_haplotypes,
        edit_pair,
        learning_rate,
        num_steps,
        max_phasing_times,
        stranded,
        out_dir
    )

    # phasing
    reads = futils.load_gz_pickle(os.path.join(obj_dir, f'reads/{chrom}_reads.pkl.gz'))
    sites = futils.load_gz_pickle(os.path.join(obj_dir, f'sites/{chrom}_sites.pkl.gz'))
    reads = phaser.get_haplos(germline_chrom, reads, sites)
    futils.save_gz_pickle(os.path.join(obj_dir, f'reads/{chrom}_reads.pkl.gz'), reads)
    phaser.save_out()


class Phaser:
    def __init__(
            self,
            chrom,
            num_haplotypes, 
            edit_pair, 
            learning_rate,
            num_steps, 
            max_phasing_times,
            stranded,
            out_dir
        ):
        self.chrom = chrom # it's for one chromosome!
        self.n_haps = num_haplotypes
        self.model = phasing_model
        self.n_steps = num_steps
        self.learning_rate = learning_rate
        self.max_phasing_times = max_phasing_times

        self.base_map = {'A':0, 'T':1, 'C':2, 'G':3}
        self.base_map_rev = dict((n, b) for b, n in self.base_map.items())
        self.rev_comp_bases = REV_COMP_BASES
        self.n_bases = len(self.base_map)
        self.rng = np.random.default_rng()
        
        self.edit_pairs = set(tuple(sorted([self.base_map[b] for b in m.split('>')])) for m in edit_pair.split(','))
        if not stranded:
            self.edit_pairs.update([tuple(sorted([self.base_map[self.rev_comp_bases[b]] for b in m.split('>')])) for m in edit_pair.split(',')])
        self.err_set = set()
        self.good_set = set()
        self.futils = FileUtils()

        self.setup_outputs(out_dir)
    
    def get_haplos(self, germline_chrom, reads, sites):
        germline_chrom['hap_snp'] = '.'
        self.germline_df = germline_chrom
        # fetch possibly interconnected snps from position-sorted reads, no need to confirm later

        single_ctr = multi_ctr = 0
        locus_cnt = 1
        groups = germline_chrom.groupby("gene")
        for gene, gene_df in groups:
            locus = f'i_{locus_cnt}' if gene == '.' else locus_cnt
            gene = gene.split('|')[0]

            if gene_df.shape[0] == 1:
                # single snp
                site_pos = gene_df['pos'].values[0]
                site_obj = sites[(self.chrom, site_pos)]
                reads = self.process_one_snp(gene, locus, site_obj, reads)
                single_ctr += 1
            else:
                # multi snps
                locus_sites = gene_df['pos'].to_list()
                locus_reads = set([r.split('|')[0] for p in locus_sites for r in sites[(self.chrom, p)].reads])
                reads = self.process_multi_snps(gene, locus, locus_reads, locus_sites, reads, sites, gene_df)
                multi_ctr += 1
            
            locus_cnt += 1
            if locus_cnt % 100 == 0:
                logging.info(f'{locus_cnt} loci done')
        self.phasing_log.write(f'* {self.chrom}: #single snp loci={single_ctr}; #multi snps loci={multi_ctr}\n')
        return reads

    def process_one_snp(self, gene, locus, site_obj, reads):
        # site is a Site obj
        hap1, hap2 = site_obj.mismatch.split('>') # okay, if it's mismatch then only 2 haps for now
        counter = dict((str(h+1), 0) for h in range(self.n_haps))

        # add hap info to reads_chrom
        for read in site_obj.reads:
            allele = read.split('|')[-1]
            if allele == '.':
                continue
            read_hap = '1' if allele == hap1 else '2'
            counter[read_hap] += 1
            if f'loc{locus}' not in reads[read.split('|')[0]].hap:
                reads[read.split('|')[0]].hap[f'loc{locus}'] = f'{read_hap}'
        
        hap_ratio = ':'.join(str(round(c/sum(counter.values()), 3)) for c in counter.values())
        hap_cnts = ':'.join(str(c) for c in counter.values())
        locus_flag = 'S'
        row_lst = [self.chrom, locus, gene, site_obj.pos, hap1, hap2, hap_ratio, hap_cnts, locus_flag]
        row = self.futils.list2line(row_lst)
        self.hap_out.write(row)
        return reads

    def process_multi_snps(self, gene, locus, locus_reads, locus_sites, reads, sites, germline_df):
        # hap seq all on forward strand
        locus_sites = np.array(sorted(list(locus_sites)))
        n_snps = locus_sites.shape[0]

        # avoid ribosomal genes
        if len(locus_reads) >= 100000:
            msg = f'{len(locus_reads)} reads in locus {locus} on chrom {self.chrom}, probably a ribosomal gene here, turn to 1-snp processing'
            logging.info(msg)
            self.phasing_log.write(msg+'\n')
            locus_site = germline_df.loc[germline_df['total'].idxmax(), 'pos']
            site_obj = sites[(self.chrom, locus_site)]
            reads = self.process_one_snp(gene, locus, site_obj, reads)
            return reads
        
        # build read table & adjacency matrix
        read_table = []
        adj_mat = np.zeros((n_snps, n_snps))
        for read in locus_reads:
            read_obj = reads[read]
            read_to_germlines = read_obj.mismatch.keys() & locus_sites
            # add counts to adj matrix
            if len(read_to_germlines) > 1:
                site_indices = [np.where(locus_sites==g)[0][0] for g in read_to_germlines]
                tmp_root = site_indices[0]
                for i in site_indices[1:]:
                    adj_mat[tmp_root, i] += 1
            read_row_num = [self.base_map[read_obj.mismatch[s]] if s in read_obj.mismatch else np.nan for s in locus_sites]
            read_table.append(read_row_num)
        read_table = np.vstack(read_table)

        # throw away low-correlated sites, this may then introduce disconnected snps
        sub_locus_idx = 1
        filted_res = self.filter_by_corr(sites, reads, gene, locus, sub_locus_idx, read_table, locus_sites, locus_reads, adj_mat)
        reads, sub_locus_idx, read_table, locus_sites, locus_reads, adj_mat = filted_res
        
        if len(locus_sites) == 0:
            return reads
        if read_table.shape[1] == 1:
            site_obj = sites[(self.chrom, locus_sites[0])]
            reads = self.process_one_snp(gene, locus, site_obj, reads)
            return reads
        
        # decompose graph
        min_num_edge = 3
        adj_mat[adj_mat < min_num_edge] = 0
        g = Graph.Adjacency(adj_mat)
        components = g.connected_components(mode='weak')
        if len(components) > 1:
            for idx, comp in enumerate(components):
                sub_locus_id = f'{locus}.{sub_locus_idx + idx}'
                
                if len(comp) == 1:
                    # treat it as single hSNP
                    pos = locus_sites[comp[0]]
                    site_obj = sites[(self.chrom, pos)]
                    reads = self.process_one_snp(gene, sub_locus_id, site_obj, reads)
                else:
                    # phase a subset of snps
                    sub_read_table = read_table[:, comp]
                    sub_locus_sites = locus_sites[comp]
                    reads_kept = ~np.isnan(sub_read_table).all(axis=1)
                    sub_read_table = sub_read_table[reads_kept]
                    sub_locus_reads = locus_reads[reads_kept]
                    reads = self.multisnp_phasing(gene, sub_locus_id, sub_read_table, sub_locus_sites, sub_locus_reads, reads)
        else:
            # phase all snps
            reads = self.multisnp_phasing(gene, locus, read_table, locus_sites, locus_reads, reads)
        return reads

    def filter_by_corr(self, sites, reads, gene, locus, locus_idx, read_table, locus_sites, locus_reads, adj_mat):
        # simple filters: coverage & unique bases
        #coverages = np.sum(~np.isnan(read_table), axis=0)
        #uniq_bases_per_col = np.array([len(np.unique(col[~np.isnan(col)])) for col in read_table.T])

        # get correlation matrix
        n_snps = read_table.shape[1]
        table = pd.DataFrame(read_table)
        corr_mat = np.abs(table.corr())
        self.phasing_log.write(str(corr_mat)+'\n')
        
        # remove columns without any good correlation with others
        ## NOTE: if it's single uncorrelated -> S
        ## if it's 2 snps uncorrelated -> 2 go to S (not U)
        ## if it's 3+ snps uncorrelated -> randomly picked 3 go to S, others U
        min_corr = 0.8 - 0.005
        num_non_na = np.sum(~np.isnan(corr_mat), axis=0)
        low_corr_vec0 = num_non_na - np.sum(corr_mat<(min_corr), axis=0)
        low_corr_vec = low_corr_vec0 == 1 # itself
        
        if sum(low_corr_vec) > 0:
            rm_site_idx = np.where(low_corr_vec)[0]
            
            multi_low_corr = low_corr_vec0 >= 2
            if np.sum(multi_low_corr) >= 3: # should be..
                # >=3 uncoorelates snps linked together
                multi_low_idx = np.where(multi_low_corr)[0]
                n_become_s = 3
                low2s = list(np.random.choice(multi_low_idx, n_become_s, replace=False))
            else:
                low2s = []
                
            if len(rm_site_idx) > 0:
                # print out
                kept_site_idx = [i for i in range(n_snps) if i not in rm_site_idx]
                kept_sites = locus_sites[kept_site_idx]
                err_sites = locus_sites[rm_site_idx]
                
                for idx, s in zip(rm_site_idx, err_sites):
                    s = int(s)
                    n_non_NA = num_non_na[idx]
                    if n_non_NA in [1, 2] or idx in low2s:
                        sub_locus_id = f'{locus}.{locus_idx}'
                        site_obj = sites[(self.chrom, s)]
                        reads = self.process_one_snp(gene, sub_locus_id, site_obj, reads)
                        locus_idx += 1
                        self.germline_df.loc[self.germline_df['pos']==s, 'hap_snp'] = 'S'
                    else:
                        self.germline_df.loc[self.germline_df['pos']==s, 'hap_snp'] = 'U'
                        
                err_cnt = self.germline_df[self.germline_df['pos'].isin(err_sites)]
                self.phasing_log.write('site(s) with low correlation: ' + str(err_cnt) + '\n')
                self.err_cnt += len(rm_site_idx) 

                # remove uncorrelated sites from read table
                self.phasing_log.write(f'removed indices: {list(map(int,rm_site_idx))}\n')
                adj_mat = adj_mat + adj_mat.T - np.diag(adj_mat.diagonal())
                read_table = read_table[:, kept_site_idx]
                kept_read_idx = ~np.isnan(read_table).all(axis=1)
                read_table = read_table[kept_read_idx]
                locus_sites = kept_sites # otherwise same locus_sites from inputs
                locus_reads = np.array(list(locus_reads))[kept_read_idx]
                adj_mat = adj_mat[:, kept_site_idx]
                adj_mat = adj_mat[kept_site_idx, :]
        return reads, locus_idx, read_table, locus_sites, np.array(list(locus_reads)), adj_mat
    
    def multisnp_phasing(self, gene, locus, read_table, locus_sites, locus_reads, reads):
        self.phasing_log.write(str(read_table)+'\n')
        phasing_time = 0
        while phasing_time < self.max_phasing_times:
            seed = np.random.randint(low=0, high=10000, size=1)[0]
            read_table = torch.from_numpy(read_table)
            data = (read_table, self.n_haps, self.n_bases)
            if phasing_time > 0 and read_table.shape[0] > 5000:
                if not np.isnan(svi_res[-1][-1]): # final loss of losses..
                    n_steps += 200
            else:
                n_steps = self.n_steps
            svi_res = phasing_svi(data, self.model, n_steps, self.learning_rate, random_seed=seed)
            
            read_table = read_table.detach().numpy()
            hap_info = self.parse_model_res(svi_res, read_table)
            hap_ratio, hap_seq, hap_assignments, hap_cnts, min_max_probs = hap_info
            # check strangeness
            locus_flag, weird_snps = self._check_weirdness(
                read_table, locus_sites, phasing_time, hap_seq, hap_ratio, min_max_probs
            )
            
            self.phasing_log.write(f'phasing time={phasing_time+1}; locus_flag={locus_flag}\n')
            if locus_flag == 'MP':
                for i, read in enumerate(locus_reads):
                    if f'loc{locus}' not in reads[read].hap:
                        reads[read].hap[f'loc{locus}'] = f'{hap_assignments[i]}'

                for s in locus_sites:
                    self.germline_df.loc[self.germline_df['pos']==s, 'hap_snp'] = 'P' # phasable
                if len(weird_snps) > 0:
                    for s in weird_snps:
                        self.germline_df.loc[self.germline_df['pos']==s, 'hap_snp'] = 'W' # weird but phasable
                break
            else:
                phasing_time += 1
        if locus_flag == 'MU':
            self.unphasables += 1
            for s in locus_sites:
                self.germline_df.loc[self.germline_df['pos']==s, 'hap_snp'] = 'U' # unphasable

        row_lst = [self.chrom, str(locus), gene, ','.join(map(str, locus_sites))] + hap_seq + [hap_ratio, hap_cnts, locus_flag]
        row = self.futils.list2line(row_lst)
        self.phasing_log.write(row)
        self.hap_out.write(row)
        return reads

    def parse_model_res(self, model_res, read_table):
        # model_res = params+losses from the model
        hap_ratio, base_probs, losses = model_res
        self.phasing_log.write(f'final loss = {losses[-1]}\n')
        hap_ratio = hap_ratio / np.sum(hap_ratio, axis=0)
        ratio = ':'.join(["%.3f" % a for a in hap_ratio])
        hap_assignments = predict_haplotypes(read_table, hap_ratio, base_probs)
        hap_cnts = ':'.join([str(sum(hap_assignments==h)) for h in range(self.n_haps)])
        
        hap_seq = []
        max_probs = []
        for h in range(self.n_haps):
            probs = []
            seqs = []
            for j in range(base_probs.shape[1]):
                base = np.argmax(base_probs[h,j,:])
                prob_res = f'hap{h} base={str(base)}; max prob={"%.4f" % max(base_probs[h,j,:])} from {list(map(lambda x: "%.4f" % x, base_probs[h,j,:]))}'
                self.phasing_log.write(f'{prob_res}\n')
                probs.append(max(base_probs[h,j,:]))
                seqs.append(self._num2base(base))
            hap_seq.append(''.join(seqs))
            max_probs.append(probs)
        min_max_probs = np.min(np.array(max_probs), axis=0)
        hap_assignments = [f'{h+1}' if h != -1 else '.' for h in hap_assignments]

        return ratio, hap_seq, hap_assignments, hap_cnts, min_max_probs

    def _check_weirdness(self, read_table, locus_sites, phasing_time, hap_seq, hap_ratio, min_max_probs):
        # quickly check if it's phasable or not
        locus_flag = "MU"
        weird_snps = []
        n_snps = read_table.shape[1]
        
        # phasable criteria...
        pert_cutoff = 0.75
        min_prob_cutoff = 0.75
        hap_ratio = list(map(float, hap_ratio.split(':')))
        match_bool = self._base_matches(read_table, hap_seq)
        prob_pass_bool = min_max_probs >= min_prob_cutoff
        pass0 = sum(match_bool==True) >= pert_cutoff*n_snps
        pass1 = sum(prob_pass_bool) >= pert_cutoff*n_snps
        pass2 = (max(hap_ratio) - min(hap_ratio)) <= 0.9

        if pass0 and pass1 and pass2:
            locus_flag = "MP"
            self.phasables += 1
            if phasing_time > 0:
                self.phasable_rescued += 1
            # pick up weird snps
            weird_snps = [locus_sites[i] for i,p in enumerate(prob_pass_bool) if not p]
            if len(weird_snps) > 0:
                self.err_cnt += len(weird_snps)
                self.phasing_log.write(f'weird snps: {weird_snps}\n')
        return locus_flag, weird_snps

    def _topn_from_col(self, col, n=None):
        # n = n_haps; col=column from table
        if not n:
            n = self.n_haps
        col = col[~np.isnan(col)].astype(int)
        res = Counter(col).most_common(n)
        # basically observed bases
        bases = [self._num2base(int(c[0])) for c in res]
        base_cnts = np.array([int(c[1]) for c in res])
        return bases, base_cnts

    def _edit_from_col(self, col):
        col = tuple(sorted(np.unique(col[~np.isnan(col)]).astype(int)))
        if self.edit_pairs.intersection([col]):
            return True
        return False
        
    def _num2base(self, num):
        # num - number in read table
        return self.base_map_rev[num]
    
    def _base_matches(self, read_table, hap_seq):
        col_bases, _ = np.apply_along_axis(self._topn_from_col, 0, read_table)
        col_bases = col_bases.T
        
        # boolean vector indicating if each column is matched between inferred & observed
        hap_seq_col = [[s for s in seq] for seq in hap_seq]
        hap_pairs = [set(pair) for pair in zip(*hap_seq_col)]
        match_bool = [self._match_obs_inf(hap_pairs[i], b) for i,b in enumerate(col_bases)]
        return np.array(match_bool)

    def _match_obs_inf(self, b_inferred, b_observed):
        '''bases: a set of bases, all +'''
        return True if len(b_inferred.intersection(b_observed)) == len(b_observed) else False
    
    def setup_outputs(self, out_dir):
        # setup counters
        self.err_cnt = 0
        self.phasables = 0
        self.unphasables = 0
        self.phasable_rescued = 0
        # setup output files
        self.hap_out = open(os.path.join(out_dir, f'{self.chrom}.haplo.tsv'), 'w')
        self.df_out = os.path.join(out_dir, f'{self.chrom}.germlines.tsv.gz')
        self.phasing_log = open(os.path.join(out_dir, f'{self.chrom}.phasing.log'), 'w')
        self.phasing_log.write(f'======================= {self.chrom} =======================\n')

    def save_out(self):
        self.germline_df.drop(columns=['chrom', 'pos']).to_csv(self.df_out, sep='\t', index=False, compression='gzip')
        self.phasing_log.write(f'* {self.err_cnt} erroneous mismatches\n')
        self.phasing_log.write(f'* {self.phasable_rescued} phasable loci rescued by seed change\n')
        self.phasing_log.write(f'* {self.phasables} phasable loci finally\n')
        self.phasing_log.write(f'* {self.unphasables} unphasable loci finally\n')
        self.phasing_log.close()
