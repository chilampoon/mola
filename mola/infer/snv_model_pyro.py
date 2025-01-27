import numpy as np
import pyro
import torch
from pyro.infer import SVI, JitTraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.params import param_store
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import ClippedAdam

# Jan 2025

@config_enumerate
def snv_diploid_model(data, prop_mut=0.2, error_rate=0.03):
    '''
    A diploid SNV model for somatic mutation detection
    n_haplotypes = 2
    n_bases = 2 (ref or alt)
    '''
    n_reads = data.shape[0]
    bases = data[:, 0]
    haps = data[:, 1]
    
    # mutation status -- somatic mutation or error
    z_mut_prior = pyro.sample(
        'z_mut_prior',
        dist.Dirichlet(1e-2 * torch.tensor([1.0, 1.0]))
    )
    z_mut = pyro.sample(
        'z_mut',
        dist.Categorical(z_mut_prior)
    )
    
    # which one is mutable haplotype
    z_hap_prior = pyro.sample(
        'z_hap_prior',
        dist.Dirichlet(1e-1 * torch.tensor([1.0, 1.0]))
    )
    z_hap = pyro.sample(
        'z_hap',
        dist.Categorical(z_hap_prior)
    )
    
    # which one is mutable base
    z_base_prior = pyro.sample(
        'z_base_prior',
        dist.Dirichlet(1e-1 * torch.tensor([1.0, 1.0]))
    )
    z_base = pyro.sample(
        'z_base',
        dist.Categorical(z_base_prior)
    )
    
    # mutant cell proportion
    p_mut = pyro.sample(
        'p_mut',
        dist.Beta(prop_mut*10, 10)
    )
    
    # error rate
    p_err = pyro.sample(
        'p_err',
        dist.Beta(error_rate*10, 10)
    )
    
    with pyro.plate('read', n_reads, dim=-1):
        hap_match = (haps == z_hap).float()
        p_mutant = z_mut * hap_match * p_mut
        
        z_base_float = z_base.float()
        prob_true_ALT = p_mutant * z_base_float + (1.0 - p_mutant) * (1.0 - z_base_float)
        
        prob_obs_alt = prob_true_ALT * (1.0 - p_err) + (1.0 - prob_true_ALT) * p_err
    
        pyro.sample(
            "obs_bases", 
            dist.Bernoulli(prob_obs_alt), 
            obs=bases.float()
        )
#pyro.render_model(snv_diploid_model, model_args=(data, 2), render_distributions=True, render_params=True)

def somatic_test_svi(data, 
        n_haplos,
        model=snv_diploid_model, 
        prop_mut=0.2,
        error_rate=0.03,
        random_seed=21, 
        lr=0.05, 
        n_steps=250
    ):
    pyro.set_rng_seed(int(random_seed))

    while True:
        pyro.clear_param_store()
        optim = ClippedAdam({"lr": lr, 
                            'betas': [0.8, 0.99]})
        guide = AutoDelta(poutine.block(model, expose=['p_mut', 'p_err', 'z_mut_prior', 'z_hap_prior', 'z_base_prior']))
        elbo = JitTraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(model, guide, optim, elbo)

    
        for step in range(n_steps):
            loss = svi.step(data, n_haplos=n_haplos, 
                            prop_mut=prop_mut, error_rate=error_rate)
            
            if np.isnan(loss):
                break
        
        if step == n_steps - 1:
            break
        else:
            # in case returns nan
            n_steps -= 50


    p_mut = pyro.param("AutoDelta.p_mut").detach().numpy()
    p_err = pyro.param("AutoDelta.p_err").detach().numpy()
    z_mut_prior = pyro.param("AutoDelta.z_mut_prior").detach().numpy()
    z_hap_prior = pyro.param("AutoDelta.z_hap_prior").detach().numpy()
    z_base_prior = pyro.param("AutoDelta.z_base_prior").detach().numpy()
    return p_mut, p_err, z_mut_prior, z_hap_prior, z_base_prior
