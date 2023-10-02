import jax
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(4)
import numpyro.distributions as dist
import numpy as np
import pandas as pd
from scipy.special import expit

from frugalCopyla import copula_lpdfs
from frugalCopyla.model	import CopulaModel
from frugalCopyla.diagnostics import *

def generate_static_model():
    return {
        'Z1': {'dist': dist.Normal, 'formula': {'loc': 'Z1 ~ 1', 'scale': 'Z1 ~ 1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': {}},
        'Z2': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'Z2 ~ 1'}, 'coeffs': {'probs': [0.5]}, 'link': {}},
        'X': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'X ~ Z1 + Z2'}, 'coeffs': {'probs': [0., 1., 1.]}, 'link': {'probs': jax.scipy.special.expit}},
        'Y': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'Y ~ X'}, 'coeffs': {'probs': [0., 1.]}, 'link': {'probs': jax.scipy.special.expit}},
        'copula': {
            'class': copula_lpdfs.multivar_gaussian_copula_lpdf,
            'vars': ['Z1', 'Z2', 'Y'],
            'formula': {
                'rho_Z1Y': 'cop ~ 1',
                'rho_Z2Y': 'cop ~ 1',
                'rho_Z1Z2': 'cop ~ 1'
            },
            'coeffs': {
                'rho_Z1Z2': [2*expit(0.1) - 1],            
                'rho_Z1Y': [2*expit(0.2) - 1],
                'rho_Z2Y': [2*expit(0.3) - 1],
            },
            'link': {}
            # 'misc': {}
        }
    }


def generate_dynamic_model():
    return {
        'A': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'A ~ 1'}, 'coeffs': {'probs': [0.5]}, 'link': {}},
        'U': {'dist': dist.Uniform, 'formula': {'low': 'U ~ 1', 'high': 'U ~ 1'}, 'coeffs': {'low': [0.], 'high': [1.]}, 'link': {}},
        'L': {'dist': dist.Exponential, 'formula': {'rate': 'L ~ A'}, 'coeffs': {'rate': [-0.3, 0.2]}, 'link': {'rate': jnp.exp}},
        'B': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'B ~ L + A'}, 'coeffs': {'probs': [-0.3, 0.4, 0.3]}, 'link': {'probs': jax.scipy.special.expit}},
        'Y': {'dist': dist.Exponential, 'formula': {'rate': 'Y ~ A + B'}, 'coeffs': {'rate': [0.5, -0.2, -0.3]}, 'link': {'probs': jnp.exp}},
        'copula': {
            'class': copula_lpdfs.multivar_gaussian_copula_lpdf,
            'vars': ['L', 'U', 'Y'],
            'formula': {
                'rho_LU': 'cop ~ 1',
                'rho_LY': 'cop ~ 1',
                'rho_UY': 'cop ~ 1'
            },
            'coeffs': {
                'rho_LU': [2*expit(1) - 1],            
                'rho_LY': [2*expit(0.5) - 1],
                'rho_UY': [2*expit(1) - 1],
            },
            'link': {}
            # 'misc': {}
        }
    }


def run_samples(num_samples, model):
    mod = CopulaModel(model)
    sim_data = mod.simulate_data(
        num_warmup=1000,
        num_samples=num_samples,
        joint_status='continuous',
        num_chains=4,
        seed=0
    )
    
    # Check sampler diagnostics
    assert check_rhat(sim_data['model'])
    min_ess = check_ess(sim_data['model'])
    subselect_samples_idx = int(np.ceil((4*num_samples)/min_ess))
    
    return pd.DataFrame(
        sim_data['data']
    )[::subselect_samples_idx]


def main():
    num_samples = 4000
    static_model = generate_static_model()
    dynamic_model = generate_dynamic_model()
    static_samples = run_samples(num_samples, static_model)
    dynamic_samples = run_samples(num_samples, dynamic_model)

    static_samples[['Z1', 'Z2', 'X', 'Y']].to_csv('static_samples.csv', index=False)
    dynamic_samples[['A', 'L', 'B', 'Y']].to_csv('dynamic_samples.csv', index=False)
    return None

if __name__ == '__main__':
    main()