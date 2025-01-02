import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns

def betabinom_params(k, n, augmented=True):
    '''
    k, n are vectors
    get global parameters empirically
    getting alpha, beta here is because scipy uses them
    '''
    if augmented:
        # augment with minor/major allele counts to make it a symmetric dist
        k2 = n - k
        k = np.hstack([k, k2])
        n = np.hstack([n, n])

    a, b = estimate_params(k, n, return_ab=True)
    mu, rho = reparam_1(a, b)
    print(f'Estimated parameters: α={a}, β={b}, μ={mu}, ρ={rho}')
    return a, b

def estimate_params(k, n, return_ab):
    '''
    m, n are vectors
    '''
    params_init = np.array([1, 1])
    # MLE
    res = minimize(neglog_likelihood_ab,
                   x0 = params_init,
                   args = (k, n),
                   method = 'L-BFGS-B',
                   bounds = [(0.001, None), (0.001, None)])
    α, β = res.x
    if return_ab:
        return α, β
    else:
        μ, ρ = reparam_1(α, β)
        return mu, rho

def neglog_likelihood_ab(params, *args):
    '''
    Negative log likelihood for beta-binom pdf
    - params: list for parameters to be estimated
    - args: 1d array containing data points
    '''
    a = params[0]
    b = params[1]
    k = args[0]
    n = args[1]
    logpdf = stats.betabinom.logpmf(k=k, n=n, a=a, b=b, loc=0)
    nll = -np.sum(logpdf)
    return nll

def reparam_1(α, β):
    μ = α / (α + β)
    ρ = 1 / (α + β + 1)
    return μ, ρ

def reparam_2(μ, ρ):
    α = μ * (1 - ρ) / ρ
    β = (1 - μ) * (1 - ρ) / ρ
    return α, β

def betabinom_posterior(data, weights, alphas, betas, augmented_idx=None):
    '''
    p(theta|k,n) = p(k,n|theta) * p(theta) / sum(p(k,n|theta) * p(theta))
    '''
    k = data[:, 0]
    n = data[:, 1]
    likelihoods = [stats.betabinom.pmf(k, n, alpha, beta) for alpha, beta in zip(alphas, betas)]
    if augmented_idx is not None:
        # 2 by default
        likelihoods[augmented_idx] = 2 * likelihoods[augmented_idx]
    raw_posteriors = [weight * likelihood for weight, likelihood in zip(weights, likelihoods)]
    posteriors = raw_posteriors / sum(raw_posteriors)
    return np.vstack(posteriors)

def betabinom_pval(k, n, a, b, side):
    '''
    return p value for beta binom test
    for one data point
    '''
    if side == 'right':
        p_val = 1 - stats.betabinom.cdf(k, n, a, b, loc=0)
    elif side == 'left':
        p_val = stats.betabinom.cdf(k, n, a, b, loc=0)
    elif side == 'either':
        mean = a / (a + b)
        # sf = 1 - cdf
        p_val = np.array([stats.betabinom.cdf(k_i, n_i, a, b, loc=0) if k_i/n_i <= mean else \
                 stats.betabinom.sf(k_i, n_i, a, b, loc=0) for k_i, n_i in zip(k, n)])
    else:
        raise ValueError("side should be either 'right', 'left' or 'either'")
    return p_val

def betabinom_mixture_pval(k, n, side, alphas, betas, weights):
    '''calculate p value with a mixture of beta binomials'''
    p_values = [betabinom_pval(k, n, a, b, side) for a, b in zip(alphas, betas)]
    weighted_p_values = [w * p for w, p in zip(weights, p_values)]
    return np.sum(weighted_p_values, axis=0)

def compute_bic(data, weights, alphas, betas):
    n_components = len(weights)
    n_params = 2*n_components + (n_components - 1)  # For alphas, betas, and weights (subtract 1 because the weights sum to 1)
    n_dp = data.shape[0]
    k = data[:,0]
    n = data[:,1]

    logpdf = np.log([w * stats.betabinom.pmf(k, n, a, b) for (w, a, b) in zip(weights, alphas, betas)]).sum()
    bic = -2*logpdf + n_params*np.log(n_dp)
    return bic

################### plotting functions ###################
def plot_beta_dist(a, b, n=5000, save_to=None):
    '''
    code is copied from 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    '''
    mean, var, skew, kurt = stats.beta.stats(a, b, moments='mvsk')
    print(f'mean={mean}, var={var}, skew={skew}, kurt={kurt}')

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(stats.beta.ppf(0.01, a, b),
                    stats.beta.ppf(0.99, a, b), 100)
    ax.plot(x, stats.beta.pdf(x, a, b),
            'p-', lw=3.5, alpha=0.6, label='beta pdf')

    r = stats.beta.rvs(a, b, size=n)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.4)
    ax.legend(loc='best', frameon=False)
    plt.title(f'n = {n} α={"%.3f" % a} β={"%.3f" % b}')
    if save_to:
        plt.savefig(save_to, dpi=1000)
    else:
        plt.show()
    plt.close()

def overlaid_betabinom_dist(obs, alphas, betas, weights=None, xlim=[0,0.5], 
                            ylim=[0,1000],n=3000, scaled_beta_pdf=True, 
                            inf_dist_labels=None, save_to=None):
    '''
    - obs: observations
    - alpha, beta: params of beta distribution
    - n: number of trails
    - save_to: path to save the plot
    '''
    sns.set(style='whitegrid', palette='colorblind') #, font='Verdana')
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('white')
    ax.grid(True, color='white')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_zorder(10)
    
    if obs is not None:
        counts, bins, patches = ax.hist(obs, bins='auto', alpha=0.35, 
                                        label='Observations')
    assert len(alphas) == len(betas)
    if len(alphas) == 1:
        weights = [1]
    for i in range(len(alphas)):
        n_i = int(weights[i] * n)
        if inf_dist_labels is not None:
            dist_label = inf_dist_labels[i] 
        else:
            dist_label = f'Inferred distribution {i}'

        if scaled_beta_pdf:
            x_i = np.linspace(0, 1, n_i)
            pdf_values = stats.beta.pdf(x_i, alphas[i], betas[i])
            # scale to match total count of observations
            pdf_values *= n_i * np.diff(bins)[0]
            ax.plot(x_i, pdf_values, label=dist_label)
        else:
            rvs = stats.beta.rvs(alphas[i], betas[i], size=n_i)
            ax.hist(rvs, alpha=0.5, bins='auto', label=dist_label)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.title(f'n={n} α={",".join(["%.3f" % a for a in alphas])} β={",".join(["%.3f" % b for b in betas])}',
              fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.legend(loc='best', frameon=False)

    if save_to:
        plt.savefig(save_to, dpi=1000)
    else:
        plt.show()
    plt.close()

def betabinom_mixture_qqplot(k, n, alphas, betas, weights, freq_mean=0.5, 
                             one_sided_pval=False, save_to=None):
    # do it when mismatch coverages are generally high
    freq = k / n
    #freq = freq / max(freq) # scale to 1
    n_dp = k.shape[0]
    empirical_qval = [sum(freq <= f)/n_dp if f <= freq_mean else sum(freq < f)/n_dp for f in freq]
    p_values = betabinom_mixture_pval(k, n, one_sided_pval, alphas, betas, weights)
    
    plt.plot(np.sort(p_values), np.sort(empirical_qval))
    plt.plot([0,1], [0,1], '--', lw=1.5, color="grey")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical Quantiles')
    plt.title('QQ Plot')

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()
    plt.close()
