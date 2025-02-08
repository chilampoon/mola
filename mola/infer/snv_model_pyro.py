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
def snv_diploid_model(data, error_prior, edit_prior, event_probs):
    '''
    A diploid SNV model for somatic mutation detection
    n_haplotypes = 2
    n_bases = 2 (ref or alt)
    '''
    n_reads = data.shape[0]
    bases = data[:, 0]
    haps = data[:, 1]
    
    # events - [error, somatic, editing], mostly are errors
    pi_event = pyro.sample(
        'pi_event', 
        dist.Dirichlet(torch.tensor(event_probs))
    )
    z_event = pyro.sample(
        "z_event", 
        dist.Categorical(pi_event)
    )
    
    # somatic - which one is mutable haplotype
    z_hap_prior = pyro.sample(
        'z_hap_prior',
        dist.Dirichlet(torch.tensor([1.0, 1.0]))
    )
    z_hap = pyro.sample(
        'z_hap',
        dist.Categorical(z_hap_prior)
    )
    
    # which one is mutable base
    z_base_prior = pyro.sample(
        'z_base_prior',
        dist.Dirichlet(torch.tensor([1.0, 1.0]))
    )
    z_base = pyro.sample(
        'z_base',
        dist.Categorical(z_base_prior)
    )
    
    # error rate
    p_err = pyro.sample(
        'p_err',
        dist.Beta(*error_prior)
    )
    
    # editing prior - allow editing frequencies different on two haplotypes
    p_edit_h0 = pyro.sample(
        'p_edit_h0',
        #dist.Beta(*edit_prior)
        dist.Beta(1, 1)
    )
    
    p_edit_h1 = pyro.sample(
        'p_edit_h1',
        #dist.Beta(*edit_prior)
        dist.Beta(1, 1)
    )
    
    # somatic - ~= mutant cell proportion
    p_mut = pyro.sample(
        'p_mut',
        dist.Beta(1, 1)
    )
    
    with pyro.plate('read', n_reads, dim=-1):
        hap_match = (haps == z_hap).float()
        z_base = z_base.float()
        
        # compute mutant probability based on category
        p_edit_haps = torch.where(haps==0, p_edit_h0, p_edit_h1)
        p_mismatch = torch.where(
            z_event == 1,  # somatic mutation: on one haplotype
            hap_match * p_mut,
            torch.where(
                z_event == 2,
                p_edit_haps, # RNA editing,
                0 # error
            )
        )
        
        prob_true_ALT = p_mismatch * z_base + (1 - p_mismatch) * (1 - z_base)
        prob_obs_alt = prob_true_ALT * (1 - p_err) + (1 - prob_true_ALT) * p_err
    
        pyro.sample(
            "obs_bases", 
            dist.Bernoulli(prob_obs_alt), 
            obs=bases.float()
        )
#pyro.render_model(snv_diploid_model, model_args=(data, [10, 500]), render_distributions=True, render_params=True)

def somatic_test_svi(data, 
        model,
        event_probs,
        error_prior,
        edit_prior,
        random_seed=21, 
        lr=0.05, 
        n_steps=250
    ):
    pyro.set_rng_seed(int(random_seed))
    
    while True:
        pyro.clear_param_store()
        optim = ClippedAdam({"lr": lr, 
                            'betas': [0.8, 0.99]})
        guide = AutoDelta(poutine.block(model, expose=['p_mut', 'p_err', 'p_edit_h0', 'p_edit_h1',
                                                    'pi_event', 'z_hap_prior', 'z_base_prior']))
        elbo = JitTraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(model, guide, optim, elbo)

        for step in range(n_steps):
            loss = svi.step(
                data,
                error_prior=error_prior,
                edit_prior=edit_prior,
                event_probs=event_probs
            )
            
            if np.isnan(loss):
                break
        
        if step == n_steps - 1:
            break
        else:
            # in case returns nan
            n_steps -= 50

    p_mut = pyro.param("AutoDelta.p_mut").detach().numpy()
    p_err = pyro.param("AutoDelta.p_err").detach().numpy()
    p_edit_h0 = pyro.param("AutoDelta.p_edit_h0").detach().numpy()
    p_edit_h1 = pyro.param("AutoDelta.p_edit_h1").detach().numpy()
    pi_event = pyro.param("AutoDelta.pi_event").detach().numpy()
    z_hap_prior = pyro.param("AutoDelta.z_hap_prior").detach().numpy()
    z_base_prior = pyro.param("AutoDelta.z_base_prior").detach().numpy()
    
    return p_mut, p_err, p_edit_h0, p_edit_h1, pi_event, z_hap_prior, z_base_prior
