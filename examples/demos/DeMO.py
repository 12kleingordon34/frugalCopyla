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


def generate_dynamic_continuous_model():
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


def generate_dynamic_discrete_model():
    return {
        'A': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'A ~ 1'}, 'coeffs': {'probs': [0.5]}, 'link': {}},
        'U': {'dist': dist.Uniform, 'formula': {'low': 'U ~ 1', 'high': 'U ~ 1'}, 'coeffs': {'low': [0.], 'high': [1.]}, 'link': {}},
        'L': {'dist': dist.Exponential, 'formula': {'rate': 'L ~ A'}, 'coeffs': {'rate': [-0.3, 0.2]}, 'link': {'rate': jnp.exp}},
        'B': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'B ~ L + A'}, 'coeffs': {'probs': [-0.3, 0.4, 0.3]}, 'link': {'probs': jax.scipy.special.expit}},
        'Y': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'Y ~ A + B'}, 'coeffs': {'probs': [0.5, -0.2, -0.3]}, 'link': {'probs': jax.scipy.special.expit}},
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
    continuous_model = generate_dynamic_continuous_model()
    discrete_model = generate_dynamic_discrete_model()
    continuous_samples = run_samples(num_samples, continuous_model)
    discrete_samples = run_samples(num_samples, discrete_model)

    continuous_samples[['A', 'L', 'B', 'Y']].to_csv('continuous_samples.csv', index=False)
    discrete_samples[['A', 'L', 'B', 'Y']].to_csv('discrete_samples.csv', index=False)
    return None

if __name__ == '__main__':
    main()