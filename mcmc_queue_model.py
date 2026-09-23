import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

def queue_model(arrival_counts, service_times, num_connections):
    """
    Bayesian M/M/c queue model to predict database saturation.
    
    Parameters:
    - arrival_counts: Array of observed request counts per time unit (e.g., per second).
    - service_times: Array of observed database query execution times in seconds.
    - num_connections: Integer representing the database connection pool size (c).
    """
    # Priors for arrival rate (lambda) and service rate (mu)
    # Gamma distribution ensures strictly positive rates
    lambda_ = numpyro.sample("lambda_", dist.Gamma(concentration=2.0, rate=0.1))
    mu = numpyro.sample("mu", dist.Gamma(concentration=2.0, rate=0.1))
    
    # Deterministic variable for traffic intensity (rho)
    # rho = lambda / (c * mu)
    rho = numpyro.deterministic("rho", lambda_ / (num_connections * mu))
    
    # Likelihood of observed arrivals (Poisson process)
    with numpyro.plate("N_arrivals", len(arrival_counts)):
        numpyro.sample("obs_arrivals", dist.Poisson(lambda_), obs=arrival_counts)
    
    # Likelihood of observed service times (Exponential distribution)
    # The rate parameter for the exponential distribution is mu.
    with numpyro.plate("N_service", len(service_times)):
        numpyro.sample("obs_service_times", dist.Exponential(mu), obs=service_times)

def run_inference(arrival_counts, service_times, num_connections, num_warmup=1000, num_samples=4000):
    """Executes MCMC sampling using the NUTS algorithm."""
    
    # Initialize the No-U-Turn Sampler
    kernel = NUTS(queue_model)
    mcmc = MCMC(
        kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=4,
        progress_bar=True
    )
    
    # Generate random key for JAX
    rng_key = jax.random.PRNGKey(0)
    
    # Execute MCMC
    mcmc.run(rng_key, arrival_counts, service_times, num_connections)
    
    return mcmc.get_samples()

def analyze_saturation(samples):
    """Calculates the probability of database queue saturation."""
    
    # Extract the posterior distribution of rho
    rho_samples = samples["rho"]
    
    # Saturation occurs when rho >= 1
    p_saturation = jnp.mean(rho_samples >= 1.0)
    
    print(f"Mean Traffic Intensity (rho): {jnp.mean(rho_samples):.4f}")
    print(f"95% HDI for rho: {jnp.percentile(rho_samples, jnp.array([2.5, 97.5]))}")
    print(f"Probability of Queue Saturation: {p_saturation * 100:.2f}%")
    
    return p_saturation

# Example Usage
if __name__ == "__main__":
    # Synthetic data generation: SATURATION SCENARIO (Traffic Spike)
    # Assuming 75 requests per second (deadline rush), executing in ~0.05 seconds (mu=20)
    true_lambda = 75.0
    true_mu = 20.0
    c = 3  # connection pool size
    
    key = jax.random.PRNGKey(42)
    key_arr, key_srv = jax.random.split(key)
    
    simulated_arrivals = jax.random.poisson(key_arr, true_lambda, shape=(100,))
    simulated_service_times = jax.random.exponential(key_srv, shape=(1000,)) / true_mu
    
    # Run the model
    print(f"Running MCMC simulation (lambda={true_lambda}, mu={true_mu}, connections={c})...")
    posterior_samples = run_inference(simulated_arrivals, simulated_service_times, c)
    analyze_saturation(posterior_samples)
