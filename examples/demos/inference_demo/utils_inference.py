import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
import pandas as pd
import optax

from frugalCopyla.model import CopulaModel
from frugalCopyla import copula_lpdfs


numpyro.set_host_device_count(4)

def create_didelez_model(rho_ly, is_rct=False):
    if is_rct: 
        didelez_spec = {
            'A': {
                'dist': dist.BernoulliProbs, 
                'formula': {'probs': 'A ~ 1'}, 
                'coeffs': {'probs': [0.]}, 
                'link': {'probs': jax.nn.sigmoid}
            },
            'L': {
                'dist': dist.Exponential, 
                'formula': {'rate': 'L ~ A'}, 
                'coeffs': {'rate': [-0.3, -0.2]}, 
                'link': {'rate': jnp.exp}
            },
            'B': {
                'dist': dist.BernoulliProbs, 
                'formula': {'probs': 'B ~ 1'}, 
                'coeffs': {'probs': [-0.3]}, 
                'link': {'probs': jax.nn.sigmoid}
            },
            'Y': {
                'dist': dist.Normal, 
                'formula': {'loc': 'Y ~ A + B + jnp.square(A : B)', 'scale': 'Y ~ 1'},
                'coeffs': {'loc': [-0.5, 0.2, 0.3, 0.], 'scale': [1.]},
                'link': {'loc': None, 'scale': None}
            },
            'copula': {
                'class': copula_lpdfs.multivar_gaussian_copula_lpdf,
                'vars': ['L', 'Y'],
                'formula': {'rho': 'cop ~ 1'},
                'coeffs': {'rho': [rho_ly]},
                'link': {'rho': None}
            }
        }
    else:
        didelez_spec = {
            'A': {
                'dist': dist.BernoulliProbs, 
                'formula': {'probs': 'A ~ 1'}, 
                'coeffs': {'probs': [0.]}, 
                'link': {'probs': jax.nn.sigmoid}
            },
            'L': {
                'dist': dist.Exponential, 
                'formula': {'rate': 'L ~ A'}, 
                'coeffs': {'rate': [-0.3, -0.2]}, 
                'link': {'rate': jnp.exp}
            },
            'B': {
                'dist': dist.BernoulliProbs, 
                'formula': {'probs': 'B ~ A + L + A * L'}, 
                'coeffs': {'probs': [-0.3, 0.4, 0.3, 0.]}, 
                'link': {'probs': jax.nn.sigmoid}
            },
            'Y': {
                'dist': dist.Normal, 
                'formula': {'loc': 'Y ~ A + B + jnp.square(A : B)', 'scale': 'Y ~ 1'},
                'coeffs': {'loc': [-0.5, 0.2, 0.3, 0.], 'scale': [1.]},
                'link': {'loc': None, 'scale': None}
            },
            'copula': {
                'class': copula_lpdfs.multivar_gaussian_copula_lpdf,
                'vars': ['L', 'Y'],
                'formula': {'rho': 'cop ~ 1'},
                'coeffs': {'rho': [rho_ly]},
                'link': {'rho': None}
            }
        }
    return CopulaModel(didelez_spec)


def sim_run(copula_model, inference_model, num_samples, svi_iter, lr, runs, progress_bar=False):
    final_params_list = []
    losses_list = []
    run = 1
    seed = 1
    while run <= runs:
        samples = copula_model.simulate_data(
            num_warmup=1000,
            num_samples=num_samples,
            joint_status='mixed',
            seed=seed,
            
        )
        data_dict = pd.DataFrame(samples['data'])[['A', 'B', 'L', 'Y']].rename(
            columns={'A': 'a_obs', 'B': 'b_obs', 'L': 'l_obs', 'Y': 'y_obs'}
        ).to_dict('list')
        for k in data_dict.keys():
            data_dict[k] = jnp.array(data_dict[k])
        
        results = mle_inference(data_dict, inference_model, lr, svi_iter, seed+1, progress_bar)
        delta_pct = 100 * (results.losses[-1] - results.losses[-2]) / results.losses[-1]
        print(f"Run: {run} / {runs}. Loss Pct diff: {delta_pct} %.")

        seed += 1
        if not jnp.isnan(results.losses[-1]):
            run += 1
            final_params_list.append(results.params)
            losses_list.append(results.losses)
    
    losses = np.array(losses_list)
    
    final_params_dict = {}
    for k in final_params_list[0].keys():
        final_params_dict[k] = jnp.array([d[k] for d in final_params_list])
 
    return {'params': final_params_dict, 'losses': losses}


def mle_inference(data, model, learning_rate, n_steps, seed, progress_bar=False):
    guide = autoguide.AutoDelta(model)
    adam = optax.adam(learning_rate=learning_rate)
    svi = numpyro.infer.SVI(
    	model, guide=guide, optim=adam, loss=numpyro.infer.Trace_ELBO()
    )
    rng_key = jax.random.PRNGKey(seed)
    svi_results = svi.run(rng_key, n_steps, data, progress_bar=progress_bar)
    return svi_results


def gaussian_copula_lpdf(u, v, rho):
    u_2 = jnp.square(u)
    v_2 = jnp.square(v)
    rho_2 = jnp.square(rho)
    return (
        -0.5 * jnp.log(1 - rho_2) - (
            rho_2 * (u_2 + v_2) - 2 * rho * u * v
        ) / (2 * (1 - rho_2))
    )


def obs_didelez_model_inference(data): 
    a_const_sigmoid = numpyro.sample('a_const_sigmoid',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    alpha_0 = numpyro.sample('alpha_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    alpha_a = numpyro.sample('alpha_a',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    gamma_0 = numpyro.sample('gamma_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    gamma_a = numpyro.sample('gamma_a',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    gamma_l = numpyro.sample('gamma_l',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )    
    gamma_al = numpyro.sample('gamma_al',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_0 = numpyro.sample('beta_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_a = numpyro.sample('beta_a',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_b = numpyro.sample('beta_b',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_ab = numpyro.sample('beta_ab',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    ) 
    
    A = numpyro.sample("a", dist.Bernoulli(jax.nn.sigmoid(a_const_sigmoid)), obs=data['a_obs'])
    L_mu = jnp.exp(-(alpha_0 + alpha_a * A))
    L = numpyro.sample("l", dist.Exponential(L_mu), obs=data['l_obs'])
    quantiles_L = dist.Exponential(L_mu).cdf(L)
    
    B_prob = jax.nn.sigmoid(
        gamma_0 + gamma_a * A + gamma_l * L + gamma_al * A * L
    )    
    B = numpyro.sample("b", dist.Bernoulli(B_prob), obs=data['b_obs'])

    Y_mean = beta_0 + beta_a * A + beta_b * B + beta_ab * A * B
    Y = numpyro.sample("y", dist.Normal(Y_mean, 1.), obs=data['y_obs'])
    quantiles_Y = dist.Normal(Y_mean, 1.).cdf(Y)

    
    # Choosing an arbitrary sigmoidal function for rho_ly
    rho_LY_val = numpyro.sample('rho_ly',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.interval(-1., 1.),
            batch_shape=(), 
            event_shape=()
        )
    )
    corr_param = jnp.array([
        [1., rho_LY_val],
        [rho_LY_val, 1.]
    ])
    
    # `numpyro.factor()` appears to add a log-likelihood to the sampling space
    # If you want to add the renormalising factors, we can simply add or subtract
    # log-likelihood factors similar to what we do on the line below.
    std_normal_L = dist.Normal(0, 1).icdf(quantiles_L)
    std_normal_Y = dist.Normal(0, 1).icdf(quantiles_Y)


    #jit_corr_mvg_copula = jax.jit(copula_lpdfs.corr_multivar_gaussian_copula_lpdf)
    #cop_log_prob = numpyro.factor(
    #    'cop_log_prob',
    #    jit_corr_mvg_copula(
    #        jnp.array([std_normal_L, std_normal_Y]).T,
    #        corr_param
    #    )
    #)
    cop_log_prob = numpyro.factor(
        'cop_log_prob',
        gaussian_copula_lpdf(
            std_normal_L,
            std_normal_Y,
            rho_LY_val
        )
    )


def rct_didelez_model_inference(data): 
    a_const_sigmoid = numpyro.sample('a_const_sigmoid',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    alpha_0 = numpyro.sample('alpha_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    alpha_a = numpyro.sample('alpha_a',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    gamma_0 = numpyro.sample('gamma_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_0 = numpyro.sample('beta_0',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_a = numpyro.sample('beta_a',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_b = numpyro.sample('beta_b',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    )
    beta_ab = numpyro.sample('beta_ab',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real,
            batch_shape=(), 
            event_shape=()
        )
    ) 
    
    A = numpyro.sample("a", dist.Bernoulli(jax.nn.sigmoid(a_const_sigmoid)), obs=data['a_obs'])
    L_mu = jnp.exp(-(alpha_0 + alpha_a * A))
    L = numpyro.sample("l", dist.Exponential(L_mu), obs=data['l_obs'])
    quantiles_L = dist.Exponential(L_mu).cdf(L)
    
    B_prob = jax.nn.sigmoid(gamma_0)    
    B = numpyro.sample("b", dist.Bernoulli(B_prob), obs=data['b_obs'])

    Y_mean = beta_0 + beta_a * A + beta_b * B + beta_ab * A * B
    Y = numpyro.sample("y", dist.Normal(Y_mean, 1.), obs=data['y_obs'])
    quantiles_Y = dist.Normal(Y_mean, 1.).cdf(Y)

    # Choosing an arbitrary sigmoidal function for rho_ly
    rho_LY_val = numpyro.sample('rho_ly',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.interval(-1., 1.),
            batch_shape=(), 
            event_shape=()
        )
    )
    
    # `numpyro.factor()` appears to add a log-likelihood to the sampling space
    # If you want to add the renormalising factors, we can simply add or subtract
    # log-likelihood factors similar to what we do on the line below.
    std_normal_L = dist.Normal(0, 1).icdf(quantiles_L)
    std_normal_Y = dist.Normal(0, 1).icdf(quantiles_Y)

    # cop_log_prob = numpyro.factor('cop_log_prob', gaussian_copula_lpdf(std_normal_L, std_normal_Y, rho_LY_val))
    cop_log_prob = numpyro.factor(
        'cop_log_prob',
        #multivar_gaussian_copula_lpdf(std_normal_L, std_normal_Y, rho_LY_val)
        gaussian_copula_lpdf(std_normal_L, std_normal_Y, rho_LY_val)
    )
