import jax
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(4)
import numpyro.distributions as dist
import pandas as pd
from scipy.special import expit

from frugalCopyla import copula_lpdfs
from frugalCopyla.model	import CopulaModel


def generate_copyla_model():
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


def main():
    model_template = generate_copyla_model()
    mod = CopulaModel(model_template)
    sim_data = mod.simulate_data(
        num_warmup=10000,
        num_samples=20000,
        joint_status='continuous',
        num_chains=4,
        seed=0
    )
    pd.DataFrame(
        sim_data['data']
    )[['X', 'Y', 'Z1', 'Z2']].to_csv('python_samples.csv')
    return None

if __name__ == '__main__':
    main()