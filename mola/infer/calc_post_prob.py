import os, re
import logging
from collections import defaultdict
import pandas as pd
pd.options.mode.chained_assignment = None
from mola.infer.betabinom_pyro import *
from mola.infer.betabinom_scipy import *
from scipy.stats import betabinom
import json
from mola.mutation.mismatch_site import REV_COMP_BASES

# 2023. updated in 2024.
# TODO: rethink of mouse non-dbSNP inference

# Number of A2I editing distributions
N_EDIT_DIST = {
    'pacbio': {
        'A>G': 2,
        'G>A': 1
    },
    'ont': {
        'A>G': 1,
        'G>A': 1
    },
    'illumina': {
        'A>G': 1,
        'G>A': 1
    }
}


class ProbCalculator:
    '''
    Calculate posteior probabilies of 
      - germline variants, 
      - RNA edits 
      - errors 
    for mismatches
    '''
    def __init__(self,
            tech,
            learning_rate,
            n_steps,
            min_nonerr_weight,
            n_nonsnp_comps,
            edit_pair,
            plot_xlim,
            plot_ylim,
            out_dir
        ):
        assert tech in N_EDIT_DIST, f'{self.tech} not supported, pick from {N_EDIT_DIST.keys()}'
        self.n_edit_dist = N_EDIT_DIST[tech]
        self.total_edit_comps = sum(self.n_edit_dist.values())
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_nonsnp_comps = n_nonsnp_comps # k for non-dbSNPs inference with flat prior
        self.min_nonerr_weight = min_nonerr_weight # ensure the error dist does not cover all datapoints in step 1
        self.edit_pair = edit_pair.split(',') #e.g. A>G,G>A
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim

        self.bb_out_dir = os.path.join(out_dir, 'betabinom_out')
        os.makedirs(self.bb_out_dir, exist_ok=True)
        self.plot_out_dir = os.path.join(self.bb_out_dir, 'betabinom_mixture_plots')
        os.makedirs(self.plot_out_dir, exist_ok=True)
        
        self.rev_comp_bases = REV_COMP_BASES
        self.has_rare_snps = True # mouse annotation doesn't contain common SNPs, only rare & none
        ## hardcoded params:
        self.augment_common_snps = True
        if self.augment_common_snps:
            # num of components in germline + error mixture
            self.n_germline_err_comps = 3
        else:
            self.n_germline_err_comps = 2
        self.fwd_strand_prob = 0.5

    def get_posterior_probs(self, mm_cnt_infer, mm_cnt_test, min_coverage, min_minor_cnt, min_minor_af):
        '''
        main args:
        - mm_cnt_infer: mismatch counts for inference (stranded)
        - mm_cnt_test: mismatch counts for testing (unstranded)
        NOTE if reads are stranded, infer and test are the same count table.
        - min_coverage: minimum coverage of sites
        - min_minor_af: minimum minor allele frequency of sites
        '''
        mm_cnt = self.filter_df(mm_cnt_infer, min_coverage, min_minor_cnt, min_minor_af)
        if mm_cnt_test is not None:
            test_mm_cnt = self.filter_df(mm_cnt_test, min_coverage, min_minor_cnt, min_minor_af)
            stranded = False
        else:
            test_mm_cnt = mm_cnt.copy()
            stranded = True
            self.min_nonerr_weight = 0

        logging.info('Inferring distribution parameters for each substitution...')

        logging.info('infer 2-comp mixtures using non-SNPs with flat prior')
        self.error_thetas = self.estimate_all_substitutions(mm_cnt, scaled_beta_pdf=False)
        
        logging.info('infer germline and editing distributions')
        self.germline_thetas, self.edit_thetas = self.get_germline_edit_params(mm_cnt)
        
        if self.has_rare_snps:
            logging.info('recalculate weights for rare snps')
            self.rare_weights = self.update_weights(mm_cnt[(mm_cnt['snp_type']=='rare')])

        logging.info('calculate posteriors for each mismatch type...')
        pos_probs_col = ['germline_post']+['error_post']+[f'edit{i+1}_post' for i in range(self.total_edit_comps)] #fixed internal order
        self.edit_thetas_for_posterior()
        for dbsnp, snp_cnt in test_mm_cnt.groupby("snp_type"):
            mm_grps = snp_cnt.groupby("mismatch")
            for mismatch, cnt_df in mm_grps:
                data = cnt_df.loc[:,['minor_cnt', 'total']].to_numpy()
                pos_probs = self.calculate_posteriors(data, mismatch, dbsnp, stranded)
                for idx, p in enumerate(pos_probs_col):
                    test_mm_cnt.loc[cnt_df.index, p] = [float("%.3f" % prob) if prob > 1e-6 else 0 for prob in pos_probs[idx, :]]
        
        # output files
        test_out = os.path.join(self.bb_out_dir, 'sites_posterior_probs.tsv.gz')
        new_col_order = np.array(test_mm_cnt.columns)
        new_col_order = list(new_col_order[new_col_order != "error_post"]) + ["error_post"]
        test_mm_cnt = test_mm_cnt[new_col_order]
        # add category column
        prob_cols = [col for col in test_mm_cnt.columns if col.endswith('_post')]
        cats = test_mm_cnt[prob_cols].idxmax(axis=1)
        cats = cats.str.replace('_post', '').str.replace('\d+', '', regex=True)
        test_mm_cnt['category'] = cats
        logging.info(f"Category value counts:\n{test_mm_cnt['category'].value_counts().to_string()}")
        
        test_mm_cnt.to_csv(test_out, sep='\t', index=False, header=True)
        param_out = os.path.join(self.bb_out_dir, 'betabinom_params.txt')
        self.print_params(param_out)

    def filter_df(self, df_file, min_coverage=None, min_minor_cnt=None, min_minor_af=None):
        df = pd.read_csv(df_file, sep='\t') # no strand
        if not df['snp_type'].str.contains('common').any():
            # for mouse snp annotation only has 'rare' & 'none' where rare means snp from db
            self.has_rare_snps = False
            df.loc[df['snp_type'] == 'rare', 'snp_type'] = 'common'
        
        if min_coverage is not None:
            df = df[df['total'] >= min_coverage]
        if min_minor_cnt is not None:
            df = df[df['minor_cnt'] >= min_minor_cnt]
        if min_minor_af is not None:
            df = df[df['minor_af'] >= min_minor_af]
        return df

    def estimate_all_substitutions(self, mm_cnt, scaled_beta_pdf):
        '''
        for all mismatch types (non-snps) estimate 2-component mixture model
        '''
        all_substitutions = mm_cnt['mismatch'].unique()
        thetas = defaultdict(dict)
        for mm in sorted(all_substitutions):
            logging.info(f'{mm} non SNP mismatches:')
            mask = (mm_cnt['mismatch'] == mm) & (mm_cnt['snp_type']=='none')
            mm_df = mm_cnt[mask]
            mm_data = mm_df.loc[:,['minor_cnt','total']].to_numpy()
            mm_data = torch.from_numpy(mm_data)
            
            while True:
                seed = int(np.random.randint(low=0, high=10000, size=1)[0])
                theta = betabinom_mixture_svi(
                    data=mm_data, 
                    n_components=self.n_nonsnp_comps,
                    random_seed=seed, 
                    model=betabinom_mixture_model,
                    lr=self.learning_rate,
                    n_steps=self.n_steps
                )
                sorted_theta = self.sort_mixture_params(theta[:-1])
                if mm not in self.edit_pair or sorted_theta[0][1] >= self.min_nonerr_weight:
                    # first comp is error, second onwards are nonerrors
                    break
                else:
                    logging.debug(f'No edit distribution detected, re-infer {mm} mixture...')
            logging.info(f'final loss = {theta[-1]}')
            thetas[mm]['weights'] = sorted_theta[0]
            thetas[mm]['alphas'] = sorted_theta[1]
            thetas[mm]['betas'] = sorted_theta[2]
            thetas[mm]['n'] = mm_df.shape[0]
            thetas[mm]['obs'] = mm_df['minor_af'].to_numpy()

        rows = len(thetas) // 3
        sns.set(style='whitegrid', palette='colorblind')
        fig, axs = plt.subplots(rows, 3, figsize=(16, 5*rows))
        for i, (mm, theta) in enumerate(thetas.items()):
            ax = axs[i//3, i%3]  # get the current axes
            ax.set_facecolor('white')
            ax.grid(True, color='white')
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_zorder(10)

            obs = theta['obs']
            alphas, betas = theta['alphas'], theta['betas']
            weights = theta['weights']
            n = round(theta['n'], -2)

            counts, bins, patches = ax.hist(obs, alpha=0.35, bins='auto', label='Obs')
            for j in range(len(alphas)):
                n_j = int(weights[j] * n)
                x_j = np.linspace(0, 1, n_j)
                if scaled_beta_pdf:
                    pdf_values = stats.beta.pdf(x_j, alphas[j], betas[j])
                    # scale to match total count of observations
                    pdf_values *= n_j * np.diff(bins)[0]
                    ax.plot(x_j, pdf_values, label=f'Inferred dist {j+1}')
                else:
                    rv = stats.beta.rvs(alphas[j], betas[j], size=n_j)
                    ax.hist(rv, alpha=0.4, bins='auto', label=f'Inferred dist {j+1}')
                ax.tick_params(axis='both', which='major', labelsize=14)

            ax.set_title(f'{mm} n={n} α={",".join(["%.2f" % a for a in alphas])} β={",".join(["%.2f" % b for b in betas])}',
                        fontsize=14)
            ax.set_ylim(self.plot_ylim)
            ax.set_xlim(self.plot_xlim)
            ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_out_dir, 'nonSNP_mixture.pdf'), dpi=1440)
        plt.close()
        return thetas

    def get_germline_edit_params(self, mm_cnt):
        gl_plot_out_dir = os.path.join(self.plot_out_dir, 'germlines')
        ed_plot_out_dir = os.path.join(self.plot_out_dir, 'edits')
        os.makedirs(gl_plot_out_dir, exist_ok=True); os.makedirs(ed_plot_out_dir, exist_ok=True)

        mismatch_types = mm_cnt['mismatch'].unique()
        germline_thetas = defaultdict(dict)
        edit_thetas = defaultdict(dict)
        for mismatch_pair in sorted(mismatch_types):
            # 1. estimate parameters for edits + errors (cz edits mixed with errors now)
            if mismatch_pair in self.edit_pair:
                logging.info(f'{mismatch_pair} editing:')
                edit_thetas[mismatch_pair] = self.infer_edit_params(mm_cnt, mismatch_pair, ed_plot_out_dir)

            # 2. estimate parameters for germlines + errors
            logging.info(f'{mismatch_pair} germline:')
            common_mask = (mm_cnt['mismatch'] == mismatch_pair) & (mm_cnt['snp_type']=='common')
            common_cnt = mm_cnt[common_mask]
            germline_thetas[mismatch_pair] = self.infer_germlines(
                mismatch_pair, edit_thetas, common_cnt, self.augment_common_snps, gl_plot_out_dir
            )

        param_set = np.unique([f'{round(a,2)}_{round(b,2)}' for t in edit_thetas.values() for a,b in zip(t['alphas'][1:], t['betas'][1:]) if a>0.1 and b>0.1])
        self.edit_param_idx = {param: idx for idx, param in enumerate(param_set)}
        return germline_thetas, edit_thetas

    def infer_edit_params(self, mm_cnt, mismatch_pair, plot_out_dir):
        compl_pair = '>'.join([self.rev_comp_bases[base] for base in mismatch_pair.split('>')])
        edit_nonsnp = mm_cnt[(mm_cnt['mismatch'] == mismatch_pair) & \
                            (mm_cnt['snp_type']=='none')]
        compl_nonsnp = mm_cnt[(mm_cnt['mismatch'] == compl_pair) & \
                            (mm_cnt['snp_type']=='none')]
        
        # extract params for complement distribution
        err_weights = self.error_thetas[compl_pair]['weights']
        err_alphas = self.error_thetas[compl_pair]['alphas']
        err_betas = self.error_thetas[compl_pair]['betas']
        ## get a plot
        overlaid_betabinom_dist(
            obs=compl_nonsnp['minor_af'],
            alphas=err_alphas,
            betas=err_betas,
            weights=err_weights,
            xlim=self.plot_xlim, 
            ylim=self.plot_ylim,
            n=round(compl_nonsnp.shape[0], -2),
            save_to=os.path.join(plot_out_dir, f'{compl_pair.replace(">", "_to_")}.mixed_betabinom.pdf')
        )

        ## usually it's the max one from non-snps, unless it's not.
        err_weight, err_alpha, err_beta = self.max_weight_params((err_weights, err_alphas, err_betas))

        # estimate edit mixtures, one of which is fixed (error)
        edit_data = edit_nonsnp.loc[:, ['minor_cnt', 'total']].to_numpy()
        edit_data = torch.from_numpy(edit_data)
        seed2 = int(np.random.randint(low=0, high=10000, size=1)[0])
        err_weight_ed = (err_weight * compl_nonsnp.shape[0]) / edit_nonsnp.shape[0]
        n_edit_components = self.n_edit_dist[mismatch_pair] + 1 # 1 is the error (from compl dist)

        edit_theta = betabinom_mixture_svi(
            data=edit_data,
            n_components=n_edit_components,
            model=betabinom_mixture_model,
            fixed_weights=[err_weight_ed],
            fixed_alphas=[err_alpha],
            fixed_betas=[err_beta],
            random_seed=seed2,
            lr=self.learning_rate,
            n_steps=self.n_steps
        )
        logging.info(f'final loss = {edit_theta[-1]}')
        weights, alphas, betas = self.sort_mixture_params(edit_theta[:-1])
        # TODO: make sure estimate correct number of edit distributions?

        # put the error paramset to the first place
        ed_mask = np.array([round(w, 2) != round(err_weight_ed, 2) for w in weights])
        edit_weights, edit_alphas, edit_betas = weights[ed_mask], alphas[ed_mask], betas[ed_mask]
        edit_weights = np.insert(edit_weights, 0, err_weight_ed)
        edit_alphas = np.insert(edit_alphas, 0, err_alpha)
        edit_betas = np.insert(edit_betas, 0, err_beta)
        
        overlaid_betabinom_dist(
            obs=edit_nonsnp['minor_af'], 
            alphas=edit_alphas, 
            betas=edit_betas, 
            weights=edit_weights, 
            xlim=self.plot_xlim, 
            ylim=self.plot_ylim, 
            n=round(edit_nonsnp.shape[0], -2), 
            save_to=os.path.join(plot_out_dir, f'{mismatch_pair.replace(">", "_to_")}.edit_mixture.pdf')
        )
        theta_dict = dict((i,j) for i,j in zip(['weights', 'alphas', 'betas'], [edit_weights, edit_alphas, edit_betas]))
        return theta_dict

    def infer_germlines(self, mismatch_pair, edit_thetas, common_cnt, augment_data, plot_out_dir):
        '''
        We only use common SNPs to do inference.
        this distribution = germlines + errors
        '''
        k = common_cnt['minor_cnt'].to_numpy()
        n = common_cnt['total'].to_numpy()

        # get error params from estimations
        if mismatch_pair in self.edit_pair:
            err_alpha = edit_thetas[mismatch_pair]['alphas'][0]
            err_beta = edit_thetas[mismatch_pair]['betas'][0]
        else:
            err_alphas = self.error_thetas[mismatch_pair]['alphas']
            err_betas = self.error_thetas[mismatch_pair]['betas']
            err_weights = self.error_thetas[mismatch_pair]['weights']
            _, err_alpha, err_beta = self.max_weight_params((err_weights, err_alphas, err_betas))

        if augment_data:
            # major + minor allele
            data = torch.tensor(np.vstack([np.array([k, n]).T, np.array([n-k, n]).T]))
            fixed_alphas = [err_alpha, err_beta]
            fixed_betas = [err_beta, err_alpha]
        else:
            data = torch.tensor(np.array([k, n]).T)
            fixed_alphas = [err_alpha]
            fixed_betas = [err_beta]

        seed = int(np.random.randint(low=0, high=10000, size=1)[0])
        theta = betabinom_mixture_svi(
            data, 
            n_components=self.n_germline_err_comps, # minor error + major error + major+minor
            model=betabinom_mixture_model, 
            fixed_alphas=fixed_alphas, 
            fixed_betas=fixed_betas, 
            random_seed=seed,
            lr=self.learning_rate, 
            n_steps=self.n_steps
        )
        logging.info(f'final loss = {theta[-1]}')
        # remove error from major allele
        if self.augment_common_snps:
            weights, alphas, betas = self.rm_augmented_param(self.sort_mixture_params(theta[:-1])) # germline + error
        else:
            weights, alphas, betas = self.sort_mixture_params(theta[:-1])
        overlaid_betabinom_dist(
            obs=common_cnt['minor_af'].to_numpy(),
            alphas=alphas, 
            betas=betas, 
            weights=weights,
            xlim=self.plot_xlim, 
            ylim=self.plot_ylim,
            n=round(common_cnt.shape[0], -2), 
            save_to=os.path.join(plot_out_dir, f'{mismatch_pair.replace(">", "_to_")}.germline_mixture.pdf')
        )
        return dict((i,j) for i,j in zip(['weights','alphas','betas'], (weights, alphas, betas)))

    def sort_mixture_params(self, param_set):
        # larger weight first
        weights, alphas, betas = param_set
        descending_order = np.argsort(weights)[::-1]
        weights = weights[descending_order]
        alphas = alphas[descending_order]
        betas = betas[descending_order]
        return weights, alphas, betas
    
    def max_weight_params(self, param_set):
        weights, alphas, betas = param_set
        max_idx = np.argmax(weights)
        return weights[max_idx], alphas[max_idx], betas[max_idx]

    def rm_augmented_param(self, germline_params):
        # remove the extra error distribution (from augmented data)
        # no need to sort beforehand
        weights, alphas, betas = germline_params
        means = alphas / (alphas + betas)
        sorted_indices = np.argsort(means)[::-1]
        
        # double the weight of the lowest mean
        lowest_mean_idx = sorted_indices[-1]
        weights[lowest_mean_idx] *= 2

        # get the weight with the second highest mean
        second_highest_idx = sorted_indices[-2]
        weights[second_highest_idx] = 1 - weights[lowest_mean_idx]

        # remove the one with the highest mean
        rm_idx = sorted_indices[0]
        weights = np.delete(weights, rm_idx)
        alphas = np.delete(alphas, rm_idx)
        betas = np.delete(betas, rm_idx)
        return weights, alphas, betas

    def likelihood_ratio(self, data, weights, alphas, betas):
        # order of params: 1) noise; 2) edit
        n = data[:, 1]
        k = data[:, 0]

        # weighted likelihood
        llh_1 = betabinom.pmf(k, n, alphas[0], betas[0]) * weights[0]
        llh_2 = betabinom.pmf(k, n, alphas[1], betas[1]) * weights[1]
        llh_ratios = llh_1 / llh_2
        # decisions = [1 if r > 1 else 0 for r in llh_ratios] # 1 = edit, = 0 noise
        return llh_ratios

    def update_weights(self, rare_cnt):
        '''weight update for rare SNPs'''
        rare_plot_dir = os.path.join(self.plot_out_dir, 'rares')
        os.makedirs(rare_plot_dir, exist_ok=True)

        # all alphas and betas are fixed, just recalculate weights
        rare_weights = defaultdict(dict)
        # reestimate weights for germline + error
        for mismatch_pair in self.germline_thetas: 
            if mismatch_pair in self.edit_thetas:
                continue
            logging.info(f'{mismatch_pair} rare:')
            sub_rare_cnt = rare_cnt[rare_cnt['mismatch']==mismatch_pair]
            data = sub_rare_cnt[['minor_cnt', 'total']].to_numpy()
            data = torch.tensor(data)
            gl_alpha, gl_beta = self.germline_thetas[mismatch_pair]['alphas'], self.germline_thetas[mismatch_pair]['betas']

            seed = int(np.random.randint(low=0, high=10000, size=1)[0])
            theta = betabinom_mixture_svi(
                data, 
                n_components=len(gl_alpha), 
                model=betabinom_mixture_model, 
                fixed_alphas=list(gl_alpha),
                fixed_betas=list(gl_beta),
                random_seed=seed,
                lr=self.learning_rate, 
                n_steps=self.n_steps
            )
            logging.info(f'final loss = {theta[-1]}')

            rare_gl_weights, rare_gl_alphas, rare_gl_betas = theta[:-1]
            rare_weights[mismatch_pair] = rare_gl_weights
            
            overlaid_betabinom_dist(
                obs=sub_rare_cnt['minor_af'].to_numpy(), 
                alphas=rare_gl_alphas, 
                betas=rare_gl_betas, 
                weights=rare_gl_weights, 
                xlim=self.plot_xlim, 
                ylim=self.plot_ylim,
                n=round(sub_rare_cnt.shape[0], -2),
                inf_dist_labels=['germline dist', 'error dist'],
                save_to=os.path.join(rare_plot_dir, f'{mismatch_pair.replace(">", "_to_")}.rare_mixture.pdf')
            )

        # reestimate weights for germline + error + edit
        plot_labels = ['germline dist', 'error dist'] + [f'edit dist {i}' for i in range(self.total_edit_comps)]
        for mismatch_pair in self.edit_thetas:
            logging.info(f'{mismatch_pair} rare:')
            sub_rare_cnt = rare_cnt[rare_cnt['mismatch']==mismatch_pair]
            data = sub_rare_cnt[['minor_cnt', 'total']].to_numpy()
            data = torch.tensor(data)
            compl_pair = '>'.join([self.rev_comp_bases[base] for base in mismatch_pair.split('>')])

            gl_alpha, gl_beta = self.germline_thetas[mismatch_pair]['alphas'][0], self.germline_thetas[mismatch_pair]['betas'][0]
            ed_alpha, ed_beta = self.edit_thetas[mismatch_pair]['alphas'], self.edit_thetas[mismatch_pair]['betas']
            # order is fixed: germline, error, edit
            fixed_alphas = np.insert(ed_alpha, 0, gl_alpha) #error alpha is in ed_alpha
            fixed_betas = np.insert(ed_beta, 0, gl_beta)
            compl_err_weight = rare_weights[compl_pair][1]
            fixed_err_weight = compl_err_weight * rare_cnt[rare_cnt['mismatch']==compl_pair].shape[0] / sub_rare_cnt.shape[0]
            compl_germline_weight = rare_weights[compl_pair][0]
            fixed_gl_weight = compl_germline_weight * rare_cnt[rare_cnt['mismatch']==compl_pair].shape[0] / sub_rare_cnt.shape[0]

            seed = int(np.random.randint(low=0, high=10000, size=1)[0])
            max_rareSNP_comps = self.n_edit_dist[mismatch_pair] + 1 + 1
            theta = betabinom_mixture_svi(
                data, 
                n_components=max_rareSNP_comps, 
                model=betabinom_mixture_model, 
                fixed_weights=[fixed_gl_weight, fixed_err_weight],
                fixed_alphas=list(fixed_alphas), 
                fixed_betas=list(fixed_betas),
                random_seed=seed, 
                lr=self.learning_rate, 
                n_steps=self.n_steps
            )
            logging.info(f'final loss = {theta[-1]}')
            rare_weights[mismatch_pair] = theta[0]

            overlaid_betabinom_dist(
                obs=sub_rare_cnt['minor_af'].to_numpy(),
                alphas=fixed_alphas,
                betas=fixed_betas,
                weights=theta[0], 
                xlim=self.plot_xlim, 
                ylim=self.plot_ylim,
                n=round(sub_rare_cnt.shape[0], -2),
                inf_dist_labels=plot_labels,
                save_to=os.path.join(rare_plot_dir, f'{mismatch_pair.replace(">", "_to_")}.rare_mixture.pdf')
            )
        return rare_weights

    def calculate_posteriors(self, data, mismatch_pair, snp_type, stranded):
        '''if reads are not stranded, then we need to calculate complement
        '''
        compl_pair = '>'.join([self.rev_comp_bases[base] for base in mismatch_pair.split('>')])

        if snp_type == 'common':
            # germline + err
            alphas = self.germline_thetas[mismatch_pair]['alphas']
            betas = self.germline_thetas[mismatch_pair]['betas']
            weights = self.germline_thetas[mismatch_pair]['weights']

            if stranded:
                post_prob = betabinom_posterior(data, weights, alphas, betas)
            else:
                alphas_compl = self.germline_thetas[compl_pair]['alphas']
                betas_compl = self.germline_thetas[compl_pair]['betas']
                weights_compl = self.germline_thetas[compl_pair]['weights']
                post_prob = self.posterior_from_strand(
                    data, fwd_weights=weights, fwd_alphas=alphas, 
                    fwd_betas=betas, rev_weights=weights_compl, rev_alphas=alphas_compl, 
                    rev_betas=betas_compl, augmented_idx=[0, 0]
                )
            # add 0s to edit probs
            post_prob = np.vstack([post_prob, self.total_edit_comps*[[0] * data.shape[0]]])
        elif snp_type == 'none':
            if mismatch_pair in self.edit_thetas:
                # error + edits
                alphas = self.edit_thetas[mismatch_pair]['alphas']
                betas = self.edit_thetas[mismatch_pair]['betas']
                weights = self.edit_thetas[mismatch_pair]['weights']
            else:
                # error
                alphas = np.hstack([self.error_thetas[mismatch_pair]['alphas'][0], self.total_edit_comps*[10]])
                betas = np.hstack([self.error_thetas[mismatch_pair]['betas'][0], self.total_edit_comps*[10]])
                weights = np.array([1] + self.total_edit_comps*[0]) # add 0s to edit probs
            
            if stranded:
                ## add 0s to germline (& edit) probs
                post_prob = betabinom_posterior(data, weights, alphas, betas)
            else:
                if compl_pair in self.edit_thetas:
                    alphas_compl = self.edit_thetas[compl_pair]['alphas']
                    betas_compl = self.edit_thetas[compl_pair]['betas']
                    weights_compl = self.edit_thetas[compl_pair]['weights']
                else:
                    alphas_compl = np.hstack([self.error_thetas[compl_pair]['alphas'], self.total_edit_comps*[10]])
                    betas_compl = np.hstack([self.error_thetas[compl_pair]['betas'], self.total_edit_comps*[10]])
                    weights_compl = np.array([1] + self.total_edit_comps*[0])
                post_prob = self.posterior_from_strand(
                    data, fwd_weights=weights, fwd_alphas=alphas, 
                    fwd_betas=betas, rev_weights=weights_compl, rev_alphas=alphas_compl, 
                    rev_betas=betas_compl, augmented_idx=[None, None]
                )
            # add 0s to germline probs
            post_prob = np.vstack([[[0] * data.shape[0]], post_prob])
        elif snp_type == 'rare':
            if mismatch_pair in self.edit_thetas:
                # germline + err + edit
                weights = self.rare_weights[mismatch_pair]
                alphas_e = self.edit_thetas[mismatch_pair]['alphas']
                betas_e = self.edit_thetas[mismatch_pair]['betas']
                alphas = np.hstack([self.germline_thetas[mismatch_pair]['alphas'][0], alphas_e])
                betas = np.hstack([self.germline_thetas[mismatch_pair]['betas'][0], betas_e])
            else:
                # germline + err
                weights = np.hstack([self.rare_weights[mismatch_pair], self.total_edit_comps*[0]])
                alphas = np.hstack([self.germline_thetas[mismatch_pair]['alphas'], self.total_edit_comps*[10]])
                betas = np.hstack([self.germline_thetas[mismatch_pair]['betas'], self.total_edit_comps*[10]])

            if stranded:
                post_prob = betabinom_posterior(data, weights, alphas, betas)
            else:
                if compl_pair in self.edit_thetas:
                    weights_compl = self.rare_weights[compl_pair]
                    alphas_compl_e = self.edit_thetas[compl_pair]['alphas']
                    betas_compl_e = self.edit_thetas[compl_pair]['betas']
                    alphas_compl = np.hstack([self.germline_thetas[compl_pair]['alphas'][0], alphas_compl_e])
                    betas_compl = np.hstack([self.germline_thetas[compl_pair]['betas'][0], betas_compl_e])
                else:
                    weights_compl = np.hstack([self.rare_weights[compl_pair], self.total_edit_comps*[0]])
                    alphas_compl = np.hstack([self.germline_thetas[compl_pair]['alphas'], self.total_edit_comps*[10]])
                    betas_compl = np.hstack([self.germline_thetas[compl_pair]['betas'], self.total_edit_comps*[10]])
                
                post_prob = self.posterior_from_strand(
                    data, fwd_weights=weights, fwd_alphas=alphas, 
                    fwd_betas=betas, rev_weights=weights_compl, rev_alphas=alphas_compl, 
                    rev_betas=betas_compl, augmented_idx=[0, 0]
                )
        else:
            raise ValueError(f'wrong dbSNP type: {snp_type}')
        return post_prob

    def posterior_from_strand(self, data, fwd_weights, fwd_alphas, fwd_betas,
                              rev_weights, rev_alphas, rev_betas,
                              augmented_idx=[None, None]):
        # augmented_idx for germline variants, [fwd, rev]
        fwd_post, rev_post = self.p_strand_given_cnt(data, fwd_weights, fwd_alphas, fwd_betas,
                                                rev_weights, rev_alphas, rev_betas, augmented_idx)
        post_given_fwd = betabinom_posterior(data, fwd_weights, fwd_alphas, fwd_betas, augmented_idx=augmented_idx[0])
        post_given_rev = betabinom_posterior(data, rev_weights, rev_alphas, rev_betas, augmented_idx=augmented_idx[1])
        posterior = self.posterior_given_strand(post_given_fwd, post_given_rev, fwd_post, rev_post)
        return posterior

    def p_strand_given_cnt(self, data,
                           fwd_weights, fwd_alphas, fwd_betas,
                           rev_weights, rev_alphas, rev_betas,
                           augmented_idx=[None, None]):
        '''
        for + strand, P(+|k,n) = P(k,n|+)*P(+)/P(k,n)
        same for -
        '''
        fwd_prob = self.fwd_strand_prob
        rev_prob = 1 - fwd_prob

        f_llhs = [stats.betabinom.pmf(data[:,0], data[:,1], a, b) for a, b in zip(fwd_alphas, fwd_betas)]
        if augmented_idx[0] is not None:
            f_llhs[augmented_idx[0]] = 2 * f_llhs[augmented_idx[0]]
        f_raw_post = [w * llh for w, llh in zip(fwd_weights, f_llhs)]
        f_raw_post_sum = sum(f_raw_post) # P(k,n|+)

        r_llhs = [stats.betabinom.pmf(data[:,0], data[:,1], a, b) for a, b in zip(rev_alphas, rev_betas)]
        if augmented_idx[1] is not None:
            r_llhs[augmented_idx[1]] = 2 * r_llhs[augmented_idx[1]]
        r_raw_post = [w * llh for w, llh in zip(rev_weights, r_llhs)]
        r_raw_post_sum = sum(r_raw_post) # P(k,n|-)

        p_cnt = np.array([f*fwd_prob + r*rev_prob for f,r in zip(f_raw_post_sum, r_raw_post_sum)]) # P(k,n)
        post_p_cnt_f = (f_raw_post_sum * fwd_prob) / p_cnt
        post_p_cnt_r = (r_raw_post_sum * rev_prob) / p_cnt
        return post_p_cnt_f, post_p_cnt_r
    
    def posterior_given_strand(self, post_given_fwd, post_given_rev, fwd_posterior, rev_posterior):
        '''
        P(theta|k,n) = P(theta,+|k,n) + P(theta,-|k,n)
                     = P(theta|k,n,+)*P(+|k,n) + P(theta|k,n,-)*P(-|k,n)
        '''
        posteriors = [pos_f*fwd_posterior + pos_r*rev_posterior for pos_f, pos_r in zip(post_given_fwd, post_given_rev)]
        return np.vstack(posteriors)

    def edit_thetas_for_posterior(self):
        # rewrite edit params to be in the same order for posterior calculation
        edit_thetas = defaultdict(dict)
        for mm, theta in self.edit_thetas.items():
            ## first one is error
            new_a = np.array([theta['alphas'][0]] + self.total_edit_comps*[10])
            new_b = np.array([theta['betas'][0]]+ self.total_edit_comps*[10])
            new_w = np.array([theta['weights'][0]]+ self.total_edit_comps*[0])
            if self.has_rare_snps:
                new_w_rare = np.hstack([self.rare_weights[mm][:2], self.total_edit_comps*[0]]) # first 2 are germline + error
                old_w_rare = self.rare_weights[mm][2:]
            for i, (a, b, w) in enumerate(zip(theta['alphas'][1:],theta['betas'][1:],theta['weights'][1:])):
                if f'{round(a,2)}_{round(b,2)}' not in self.edit_param_idx:
                    continue
                param_idx = self.edit_param_idx[f'{round(a,2)}_{round(b,2)}']
                new_a[param_idx+1] = a; new_b[param_idx+1] = b; new_w[param_idx+1] = w # 1 is error
                if self.has_rare_snps:
                    new_w_rare[param_idx+2] = old_w_rare[i] # 2 is germline+err
            edit_thetas[mm]['alphas'] = new_a
            edit_thetas[mm]['betas'] = new_b
            edit_thetas[mm]['weights'] = new_w
            if self.has_rare_snps:
                self.rare_weights[mm] = new_w_rare
        self.edit_thetas = edit_thetas

    def print_params(self, fout):
        with open(fout, 'w') as f:
            f.write('* Error parameters - initial:\n')
            for mm, theta in self.error_thetas.items():
                params = "; ".join([f'{k}={",".join(["%.3f" % i for i in v])}' for k,v in theta.items() if k not in ["obs","n"]])
                f.write(f'  - {mm}: {params}\n')
            f.write('\n* Germline parameters - germline + germline error:\n')
            for mm, theta in self.germline_thetas.items():
                params = "; ".join([f'{k}={",".join(["%.3f" % i for i in v])}' for k,v in theta.items()])
                f.write(f'  - {mm}: {params}\n')
            f.write('\n* Edit parameters - non-dbSNP error + editing(s):\n')
            for mm, theta in self.edit_thetas.items():
                params = "; ".join([f'{k}={",".join(["%.3f" % i for i in v])}' for k,v in theta.items()])
                f.write(f'  - {mm}: {params}\n')
            if self.has_rare_snps:
                f.write('\n* Rare SNP weights - germline + error (+ edit):\n')
                for mm, theta in self.rare_weights.items():
                    params = ",".join(["%.3f" % p for p in theta])
                    f.write(f'  - {mm}: {params}\n')

        json_out = re.sub(r'\.txt$', '.json', fout)
        with open(json_out, 'w') as j:
            j.write('error_params = \n')
            error_thetas = defaultdict(dict)
            for mm,theta in self.error_thetas.items():
                for k,v in theta.items():
                    if k in ['obs', 'n']: continue
                    if type(v) is np.ndarray:
                        error_thetas[mm][k] = v.tolist()
            json.dump(error_thetas, j, indent = 2)
            j.write('\n\ngermline_params = \n')
            germline_thetas = self._dictnp2list(self.germline_thetas)
            json.dump(germline_thetas, j, indent = 2)
            j.write('\n\nedit_params = \n')
            edit_thetas = self._dictnp2list(self.edit_thetas)
            json.dump(edit_thetas, j, indent = 2)
            if self.has_rare_snps:
                j.write('\n\nrare_weights = \n')
                rare_weights = self._dictnp2list(self.rare_weights)
                json.dump(rare_weights, j, indent = 2)
            j.write('\n')

    def _dictnp2list(self, param_dict):
        new_dict = param_dict
        for mm, theta in param_dict.items():
            if type(theta) is np.ndarray:
                new_dict[mm] = theta.tolist()
            else:
                for k, v in theta.items():
                    if type(v) is np.ndarray:
                        new_dict[mm][k] = v.tolist()
        return new_dict
