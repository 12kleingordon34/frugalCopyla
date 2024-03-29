import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import jax.scipy.stats as jax_stats
import numpyro.distributions as dist


def _reshape_matrix(n_vars: int, rhos: list[float], is_cholesky=False) -> jnp.array:
    assert int((n_vars ** 2 - n_vars) / 2) == len(rhos)
    position_idx = 0
    for n in reversed(range(1, n_vars+1)):
        rhos.insert(position_idx, 1)
        position_idx += n
    corr = jnp.zeros((n_vars, n_vars))
    triu = jnp.triu_indices(n_vars)
    tril = jnp.tril_indices(n_vars, -1)
    if is_cholesky:
        corr = corr.at[triu].set(rhos).T
        for i in range(len(corr)):
            corr = corr.at[i, i].set(jnp.sqrt(2 - corr[i, ].dot(corr[i, ])))
    else:
        corr = corr.at[triu].set(rhos).T.at[triu].set(rhos)
    return corr


def multivar_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    rvs = jnp.array(list(vars.values()))
    std_gaussian_rvs = dist.Normal(0, 1).icdf(rvs)
    cov = _reshape_matrix(len(std_gaussian_rvs), list(rhos.values()), is_cholesky=False)
    llhood = jnp.log(
        jnp.linalg.det(cov) ** (-1/2) * jnp.exp(
            -0.5 * jnp.matmul(
                std_gaussian_rvs.T,
                jnp.matmul(
                    jnp.linalg.inv(cov) - jnp.identity(len(std_gaussian_rvs)),
                    std_gaussian_rvs
                )
            )
        )
    )
    return llhood


def multivar_studentt_copula_lpdf(vars: dict, rhos: dict, df: float) -> float:
    assert df > 0
    rvs = jnp.array(list(vars.values()))
    std_studentt_rvs = dist.StudentT(df, 0, 1).icdf(rvs)
    cor = _reshape_matrix(len(std_studentt_rvs), list(rhos.values()), is_cholesky=False)
    cor_chol = jnp.linalg.cholesky(cor)
    llhood = dist.MultivariateStudentT(
        df, loc=jnp.zeros(len(std_studentt_rvs)), scale_tril=cor_chol
    ).log_prob(std_studentt_rvs).sum() - jnp.sum(dist.StudentT(df, 0, 1).log_prob(std_studentt_rvs))
    return llhood


def bivariate_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    std_gaussian_dist = dist.Normal(0, 1)
    u, v = [std_gaussian_dist.icdf(x) for x in vars.values()]
    rho = list(rhos.values())[0]
    u_2 = jnp.square(u)
    v_2 = jnp.square(v)
    rho_2 = jnp.square(rho)
    llhood = (
        -0.5 * jnp.log(1 - rho_2) - (
            rho_2 * (u_2 + v_2) - 2 * rho * u * v
        ) / (2 * (1 - rho_2))
    )
    return llhood


def independence_copula_lpdf(vars: dict, rhos: dict) -> float:
    llhood = jnp.log(1.)
    return llhood


def corr_multivar_gaussian_copula_lpdf(rvs: jnp.array, corr: jnp.array) -> float:
    llhood = (
        dist.MultivariateNormal(
           covariance_matrix=corr
        ).log_prob(rvs)
#        + 0.5 * len(rvs) * jnp.log(2 * jnp.pi)
#        + 0.5 * jnp.sum((rvs ** 2))
    )
    return llhood
