import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import math

# ==============================================================================
# ADVERSARIAL QUEUE HARNESS: EMPIRICAL VERIFIER
# Architecture: M/G/c Queue (Markov Arrivals, General/Heavy-Tailed Service)
# Purpose: Empirically falsify heuristic capacity bounds against 
#          non-memoryless (Weibull) traffic flows.
# ==============================================================================

def general_queue_model(arrival_counts, service_times, num_connections):
    lambda_ = numpyro.sample("lambda_", dist.Gamma(concentration=2.0, rate=0.1))
    
    with numpyro.plate("N_arrivals", len(arrival_counts)):
        numpyro.sample("obs_arrivals", dist.Poisson(lambda_), obs=arrival_counts)

    k = numpyro.sample("weibull_k", dist.HalfNormal(2.0))
    scale = numpyro.sample("weibull_scale", dist.HalfNormal(5.0))
    
    with numpyro.plate("N_service", len(service_times)):
        numpyro.sample("obs_service_times", dist.Weibull(scale=scale, concentration=k), obs=service_times)

    mean_service_time = scale * jax.scipy.special.gamma(1.0 + (1.0 / k))
    mu = numpyro.deterministic("mu", 1.0 / mean_service_time)
    rho = numpyro.deterministic("rho", lambda_ / (num_connections * mu))

def run_empirical_verification(arrival_counts, service_times, num_connections, proposed_bound_func):
    print("\n[VERIFIER] Ingesting non-memoryless traffic data...")
    kernel = NUTS(general_queue_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=2, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0), arrival_counts, service_times, num_connections)
    
    samples = mcmc.get_samples()
    rho_samples = samples["rho"]
    p_saturation = jnp.mean(rho_samples >= 1.0)
    
    print(f"[VERIFIER] Empirical Probability of Saturation: {p_saturation * 100:.2f}%")
    print(f"[VERIFIER] 95% HDI for Traffic Intensity (rho): {jnp.percentile(rho_samples, jnp.array([2.5, 97.5]))}")
    
    is_stable_according_to_bound = proposed_bound_func(rho_samples)
    
    if is_stable_according_to_bound and p_saturation > 0.05:
        print("\n[!] REJECTION: Proposed bound violated thermodynamic reality.")
        print("[!] Feedback to Generator: False Positive on Stability. Mutate bound.")
        return False
    elif not is_stable_according_to_bound and p_saturation < 0.01:
        print("\n[!] REJECTION: Proposed bound is overly restrictive (False Negative).")
        print("[!] Feedback to Generator: Mutate bound for higher capacity.")
        return False
    else:
        print("\n[+] VALIDATED: Proposed bound aligns with heavy-tailed reality.")
        return True

def llm_candidate_bound(rho_samples):
    p90_rho = jnp.percentile(rho_samples, 90.0)
    return p90_rho < 1.0

if __name__ == "__main__":
    true_lambda = 45.0
    k_true = 0.8
    scale_true = 0.05
    c = 3
    
    key = jax.random.PRNGKey(42)
    key_arr, key_srv = jax.random.split(key)
    
    simulated_arrivals = jax.random.poisson(key_arr, true_lambda, shape=(200,))
    U = jax.random.uniform(key_srv, shape=(2000,))
    simulated_service_times = scale_true * jnp.power(-jnp.log(U), 1.0 / k_true)
    
    print("=== BAYESIAN QUEUE: BOUND EVALUATION ===")
    print("Simulating Heavy-Tailed (Bursty) Traffic Dynamics...")
    
    success = run_empirical_verification(simulated_arrivals, simulated_service_times, c, llm_candidate_bound)
