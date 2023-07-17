import jax.numpy as jnp
import numpyro.distributions as dist


def bivariate_gaussian_copula_hfunction(vars: dict, rho: float) -> float:
    assert (rho >= -1.) and (rho <= 1.)

    std_gaussian_dist = dist.Normal(0, 1)
    u, v = vars.values()
    return std_gaussian_dist.cdf(
        (std_gaussian_dist.icdf(u) - rho * std_gaussian_dist.icdf(v)) / jnp.sqrt(1 - jnp.square(rho))
    )


def bivariate_student_copula_hfunction(vars: dict, rho: float, df: int) -> float:
    assert (rho >= -1.) and (rho <= 1.)
    assert df >= 0
    u, v = vars.values()

    t_df = dist.StudentT(loc=0.0, scale=1., df=df)
    t_dfp1 = dist.StudentT(loc=0.0, scale=1., df=df+1)
    if any(u) == 1:
        return jnp.array(1.)
	# Otherwise calculate conditional CDF
    else:
        return t_dfp1.cdf(
            (t_df.icdf(u) - rho * t_df.icdf(v)) / (
                jnp.sqrt(
                    (
                        df + jnp.square(t_df.icdf(v))) * (1 - jnp.square(rho)
                    ) / (df + 1)
                )
            )
        )


def independence_copula_hfunction(vars: dict, rho: float) -> float:
    u, _ = vars.values()
    return u
