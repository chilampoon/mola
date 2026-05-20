import numpy as np
import pyro
import torch
from pyro import poutine

from mola.infer.betabinom_pyro import betabinom_mixture_model


def test_betabinom_mixture_model_accepts_numpy_float32_fixed_weights():
    pyro.clear_param_store()
    data = torch.tensor([[1, 10], [2, 10]])

    trace = poutine.trace(betabinom_mixture_model).get_trace(
        data=data,
        n_components=2,
        fixed_weights=[np.float32(0.25)],
        fixed_alphas=[np.float32(1.0)],
        fixed_betas=[np.float32(10.0)],
    )

    assert "w" in trace.nodes
