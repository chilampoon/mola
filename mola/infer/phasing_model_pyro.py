"Bayesian phasing model with pyro"

import pyro
import torch
from torch.distributions import constraints
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.params import param_store
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import ClippedAdam
import numpy as np

@config_enumerate   
def phasing_model(read_table, n_haplos, n_bases):
    n_reads, n_snps = read_table.shape
    
    # mask for missing values
    is_observed = (~torch.isnan(read_table))
    valid_table = torch.nan_to_num(read_table, nan=0).to(torch.int32)
    
    # haplotype params
    with pyro.plate("hap", n_haplos, dim=-2):
        β = pyro.sample(
            "β",
            dist.Gamma(1/n_haplos, 1)
        )
        with pyro.plate("pos", n_snps):
            concentration = 1e-10
            ω = pyro.sample(
                "ω", 
                dist.Dirichlet(concentration * torch.ones(n_bases)/n_bases)
            )
            
    # read params
    with pyro.plate("read_i", n_reads, dim=-2):
        β = β.reshape(1,-1)[0]
        h = pyro.sample(
            "h_i", 
            dist.Categorical(β)
        )
        with pyro.plate("pos_j", n_snps) as j:
            pyro.sample(
                "x_ij", 
                dist.Categorical(ω[h,j,:]).mask(is_observed),
                obs=valid_table
            )
# pyro.render_model(phasing_model, model_args=(read_table,), render_distributions=True, render_params=True)            

def parameterized_guide(read_table, n_haplos, n_bases):
    '''
    please use autoguide instead
    '''
    n_reads, n_snps = read_table.shape
    
    β_q = pyro.param(
        "β_q", 
        lambda: torch.ones(n_haplos, 1),
        constraint=constraints.positive
    )
    
    ω_q = pyro.param(
        "ω_q",
        lambda: torch.ones(n_haplos, n_snps, n_bases),
        constraint=constraints.positive
    )

    with pyro.plate("hap", n_haplos, dim=-2) as h:
        pyro.sample("β", dist.Gamma(β_q[h], 1.0))
        with pyro.plate("pos", n_snps):
            pyro.sample("ω", dist.Dirichlet(ω_q))
    
    with pyro.plate("read_i", n_reads, dim=-2) as r:
        h = pyro.sample(
            "h_i", 
            dist.Categorical(β_q.reshape(1,-1)[0])
        )

def phasing_svi(data, model, n_steps, learning_rate, random_seed, loss_cutoff=30):
    pyro.set_rng_seed(int(random_seed))
    optim = ClippedAdam({"lr": learning_rate, 
                         'betas': [0.85, 0.99]})
    read_table, n_haplos, n_bases = data
    
    while True:
        pyro.clear_param_store()
        guide = AutoDelta(poutine.block(model, expose=['β', 'ω']))
        elbo = JitTraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(model, guide, optim, elbo)

        losses = []
        for step in range(n_steps):
            loss = svi.step(read_table, n_haplos=n_haplos, n_bases=n_bases)
            losses.append(loss)
            # if step % 100 == 0:
            #     print("Elbo loss: {}".format(loss))
            if loss < loss_cutoff or np.isnan(loss):
                break
        
        if step == n_steps - 1 or losses[-1] < loss_cutoff:
            break
        else:
            # in case returns nan
            n_steps -= 200
    
    β = pyro.param("AutoDelta.β").detach().numpy().reshape(1,-1)[0]
    ω = pyro.param("AutoDelta.ω").detach().numpy()
    return β, ω, losses

def predict_haplotypes(read_table, beta_normed, omega):
    # infer h_i for read_i
    if type(read_table) is torch.Tensor:
        read_table = read_table.detach().numpy()
    
    h_posteriors = np.apply_along_axis(h_i_posterior, 1, read_table, 
                                       beta_normed, omega)
    h_posteriors = h_posteriors/np.sum(h_posteriors, axis=1, keepdims=True)
    max_h_posteriors = np.max(h_posteriors, axis=1)
    # mask reads with vague haplotypes
    vague_mask = [(np.isnan(p)) or (p <= 0.6 and p >= 0.4) for p in max_h_posteriors]
    haplos = np.argmax(h_posteriors, axis=1)
    haplos[vague_mask] = -1
    return haplos

def h_i_posterior(read_i, beta, omega):
    # P(h_i|x) = P(x|h_i)*P(h_i) / P(x)
    n_haplos = omega.shape[0]
    posteriors = np.array([])
    for k in range(n_haplos):
        likelihood = np.prod([omega[k,j,int(x)] for j,x in enumerate(read_i) if ~np.isnan(x)])
        prior = beta[k]
        posterior_raw = likelihood * prior
        posterior_raw = 1e-20 if np.isnan(posterior_raw) else posterior_raw
        posteriors = np.append(posteriors, posterior_raw)
    return posteriors
