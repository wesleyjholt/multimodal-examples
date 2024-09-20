import numpy as np
import numpyro

def eval_numpyro_model(model, *model_args, seed=None, **model_kwargs):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    with numpyro.handlers.seed(rng_seed=seed):
        return model(*model_args, **model_kwargs)