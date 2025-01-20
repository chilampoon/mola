import numpy as np
import pyro
import torch
from pyro.infer import SVI, JitTraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.params import param_store
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import ClippedAdam

@config_enumerate
def snv_model(data, n_haplos, prop_mut=0.2, error_rate = 0.05):
    n_reads = data.shape[0]
    base = data[:, 0]
    hap = data[:, 1]
    n_bases = torch.unique(base).size(0)
    
    # sequencing error rate
    ε = pyro.sample(
        'ε', 
        dist.Beta(error_rate * 10, 10)
    )

    # params for haplotype
    with pyro.plate("hap", n_haplos, dim=-1):
        ν = pyro.sample(
            "ν",
            dist.Beta(prop_mut * 10, 10)
        )
        
        η = pyro.sample('η', dist.Gamma(1/n_haplos, 1))
    
    # params for read
    with pyro.plate('read', n_reads, dim=-1):
        η = η.reshape(1,-1)[0]
        h = pyro.sample(
            "h",
            dist.Categorical(η),
            obs=hap
        )
        # true base probs + error probs
        true_probs = torch.stack([1 - ν[h], ν[h]], dim=-1)
        # P(observed base | true base) = (1 - ε)P(true base) + εP(random base)
        error_mat = torch.eye(n_bases) * (1 - ε - ε/(n_bases-1)) + torch.ones(n_bases, n_bases) * (ε/(n_bases-1))
        
        expanded_probs = torch.zeros(n_reads, n_bases)
        expanded_probs[:, :true_probs.shape[-1]] = true_probs
        observed_probs = torch.matmul(expanded_probs, error_mat)
        
        pyro.sample(
            "x", 
            dist.Categorical(observed_probs),
            obs=base
        )

#pyro.render_model(snv_model, model_args=(data, 2), render_distributions=True, render_params=True)

def somatic_test_svi(data, 
        n_haplos,
        model=snv_model, 
        prop_mut=0.2,
        error_rate=0.05,
        random_seed=21, 
        lr=0.05, 
        n_steps=250
    ):
    pyro.set_rng_seed(int(random_seed))

    while True:
        pyro.clear_param_store()
        optim = ClippedAdam({"lr": lr, 
                            'betas': [0.8, 0.99]})
        guide = AutoDelta(poutine.block(snv_model, expose=['ν', 'η', 'ε']))
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


    hap_mut_rates = pyro.param("AutoDelta.ν").detach().numpy()
    η = pyro.param("AutoDelta.η").detach().numpy()
    hap_ratio = η / η.sum()
    error_rate = pyro.param("AutoDelta.ε").detach().numpy()
    return hap_mut_rates, hap_ratio, error_rate
