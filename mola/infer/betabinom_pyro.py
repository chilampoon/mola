import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, JitTraceEnum_ELBO, config_enumerate
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
from pyro.optim import ClippedAdam
from pyro.infer.autoguide.guides import AutoDelta


################### n-component mixture model (2023) ###################
@config_enumerate
def betabinom_mixture_model(data, 
        n_components, 
        weight_concentration=None,
        weight_prior=None, 
        fixed_weights=None, 
        fixed_alphas=None, 
        fixed_betas=None):
    k = data[:, 0]
    n = data[:, 1]
    
    if not fixed_weights and not weight_prior:
        # flat prior
        weight_concentration = 1e-8
        weights = pyro.sample(
            "weights", 
            dist.Dirichlet(weight_concentration * torch.ones(n_components)/n_components)
        )
    elif weight_prior:
        # with prior
        assert weight_concentration is not None, 'specify weight_concentration!'
        assert len(weight_prior) == n_components, 'weight_prior should have n_components elements'

        weights = pyro.sample(
            "weights",
            dist.Dirichlet(weight_concentration * torch.tensor(weight_prior))
        )
    else:
        # the first N components have fixed weights (from error distribution)
        # other components have flexible weights
        assert type(fixed_weights) is list, 'fixed_weights should be a list'
        weight_prior_oth = 1 - sum(fixed_weights)
        n_oth_components = n_components - len(fixed_weights)
        with pyro.plate("component_weights", n_oth_components, dim=-1):
            weight_other = pyro.sample(
                "w",
                dist.Gamma(weight_prior_oth / n_oth_components, 1)
            )
        flexible_weights = weight_prior_oth * weight_other / torch.sum(weight_other)
        weights = torch.cat([torch.tensor(fixed_weights, dtype=torch.float32), flexible_weights])
    with pyro.plate("components", n_components, dim=-1):
        alpha = pyro.sample(
            "alpha",
            dist.Gamma(1, 1)
        )
        beta = pyro.sample(
            "beta",
            dist.Gamma(1, 1)
        )
    
    if fixed_alphas is not None:
        # the first N parameters are fixed
        assert type(fixed_alphas) is list, 'fixed_alphas should be a list'
        for i in range(len(fixed_alphas)):
            alpha[i] = torch.tensor(fixed_alphas[i], dtype=torch.float32)

    if fixed_betas is not None:
        assert type(fixed_betas) is list, 'fixed_betas should be a list'
        for i in range(len(fixed_betas)):
            beta[i] = torch.tensor(fixed_betas[i], dtype=torch.float32)

    with pyro.plate("data", len(data), dim=-1):
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        pyro.sample(
            "obs", 
            dist.BetaBinomial(
                total_count=n, 
                concentration0=beta[assignment],
                concentration1=alpha[assignment]
            ), 
            obs=k
        )
# pyro.render_model(betabinom_mixture_model, model_args=(edit_data, 3), render_distributions=True, render_params=True)

def betabinom_mixture_svi(data, 
        n_components, 
        model=betabinom_mixture_model, 
        weight_concentration=None, 
        weight_prior=None, 
        fixed_weights=None, 
        fixed_alphas=None, 
        fixed_betas=None,
        random_seed=21, 
        lr=0.01, 
        n_steps=2000):
    pyro.set_rng_seed(random_seed)
    
    while True:
        pyro.clear_param_store()
        guide = AutoDelta(poutine.block(model, expose=['w', 'alpha', 'beta', 'weights']))
        adam = ClippedAdam({"lr": lr, "betas": [0.8, 0.99]})
        svi = SVI(model, guide, adam, loss=JitTraceEnum_ELBO(max_plate_nesting=1))
        
        for step in range(n_steps):
            loss = svi.step(data=data, n_components=n_components, 
                weight_concentration=weight_concentration, 
                weight_prior=weight_prior, fixed_weights=fixed_weights,
                fixed_alphas=fixed_alphas, fixed_betas=fixed_betas)
        
            if np.isnan(loss):
                break
        if step == n_steps - 1:
            break
        else:
            # in case returns nan
            n_steps -= 200

    if fixed_weights is not None:
        kappa = pyro.param("AutoDelta.w").detach().numpy()
        weights = np.concatenate([fixed_weights, (1-sum(fixed_weights))*kappa/sum(kappa)])
    else:
        weights = pyro.param("AutoDelta.weights").detach().numpy()
    alphas = pyro.param("AutoDelta.alpha").detach().numpy()
    betas = pyro.param("AutoDelta.beta").detach().numpy()
    return weights, alphas, betas, loss


################### single beta-binom dist ###################
def betabinom_model(data):
    # !! The positions of alpha and beta are 
    # reversed in torch.distributions.Beta !!
    # https://pytorch.org/docs/stable/_modules/torch/distributions/beta.html#Beta

    k = data[:, 0] # successes
    n = data[:, 1] # trails

    alpha = pyro.param(
        "alpha", 
        torch.tensor(1.0), 
        constraint=constraints.positive
    )
    beta = pyro.param(
        "beta", 
        torch.tensor(1.0), 
        constraint=constraints.positive
    )
    
    with pyro.plate("counts", data.size(0)):
        pyro.sample(
            "obs", 
            dist.BetaBinomial(total_count=n, 
                              concentration0=beta, 
                              concentration1=alpha), 
            obs=k
        )

def betabinom_guide(data):
    pass

@config_enumerate
def parameterized_guide(data, noise_alpha, noise_beta, K=3):
    '''
    please use autoguide instead...
    '''
    weights_q = pyro.param(
        "weights_q", 
        lambda: torch.ones(K)/K, 
        constraint=constraints.simplex
    )

    alpha_q = pyro.param(
        "alpha_q", 
        lambda: torch.tensor([noise_alpha, 1]), 
        constraint=constraints.positive
    )
    beta_q = pyro.param(
        "beta_q", 
        lambda: torch.tensor([noise_beta, 1]), 
        constraint=constraints.positive
    )

    pyro.sample("weights", dist.Dirichlet(weights_q))
    with pyro.plate("components", K) as c:
        pyro.sample("alpha", dist.Gamma(alpha_q[c], 1))
        pyro.sample("beta", dist.Gamma(beta_q[c], 1))

    with pyro.plate("data", len(data)):
        assignment_probs = pyro.param(
            "assignment_probs",
            torch.ones(len(data), K) / K,
            constraint=constraints.simplex,
        )
        pyro.sample("assignment", dist.Categorical(assignment_probs))

def betabinom_svi(data, model, guide, random_seed=21, lr=0.01, n_steps=1000):
    pyro.set_rng_seed(random_seed)
    pyro.clear_param_store()
    adam = ClippedAdam({"lr": lr, "betas": [0.8, 0.99]})
    svi = SVI(model, guide, adam, loss=JitTraceEnum_ELBO())

    for step in range(n_steps):
        loss = svi.step(data)
        if step % 50 == 0:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))

    a = pyro.param("alpha").item()
    b = pyro.param("beta").item()
    return a, b, loss
