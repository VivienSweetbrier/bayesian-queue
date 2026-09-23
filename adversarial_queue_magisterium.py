import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import math

# ==============================================================================
# SWEETBRIER MAGISTERIUM: ADVERSARIAL QUEUE HARNESS
# Architecture: M/G/c Queue (Markov Arrivals, General/Heavy-Tailed Service)
# Purpose: Empirically falsify DreamLLM-generated capacity bounds against 
#          non-memoryless (Weibull/Pareto) traffic flows.
# ==============================================================================

def general_queue_model(arrival_counts, service_times, num_connections):
    """
    Bayesian M/G/c queue model. Service times are strictly non-memoryless (Weibull).
    This simulates real-world bursty traffic, supply chain shocks, or complex queries.
    """
    # 1. Arrival Rate (Lambda) - Poisson process (Memoryless)
    lambda_ = numpyro.sample("lambda_", dist.Gamma(concentration=2.0, rate=0.1))
    
    with numpyro.plate("N_arrivals", len(arrival_counts)):
        numpyro.sample("obs_arrivals", dist.Poisson(lambda_), obs=arrival_counts)

    # 2. General Service Time (Heavy-Tailed Weibull instead of Exponential)
    # Weibull has two parameters: concentration (shape, k) and scale (lambda, lambda_w)
    k = numpyro.sample("weibull_k", dist.HalfNormal(2.0))
    scale = numpyro.sample("weibull_scale", dist.HalfNormal(5.0))
    
    with numpyro.plate("N_service", len(service_times)):
        numpyro.sample("obs_service_times", dist.Weibull(scale=scale, concentration=k), obs=service_times)

    # 3. Calculate True Service Rate (Mu) from the General Distribution
    # Mean of Weibull = scale * Gamma(1 + 1/k)
    # JAX doesn't have a direct Gamma function in standard jnp, we use jax.scipy.special.gamma
    mean_service_time = scale * jax.scipy.special.gamma(1.0 + (1.0 / k))
    mu = numpyro.deterministic("mu", 1.0 / mean_service_time)

    # 4. Traffic Intensity (Rho)
    rho = numpyro.deterministic("rho", lambda_ / (num_connections * mu))

def run_magisterium_evaluation(arrival_counts, service_times, num_connections, proposed_bound_func):
    """
    The Magisterium Verifier. 
    Executes HMC on the chaotic data, then tests DreamLLM's proposed bound.
    """
    print("\n[MAGISTERIUM] Ingesting non-memoryless traffic data...")
    kernel = NUTS(general_queue_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=2, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0), arrival_counts, service_times, num_connections)
    
    samples = mcmc.get_samples()
    rho_samples = samples["rho"]
    p_saturation = jnp.mean(rho_samples >= 1.0)
    
    print(f"[MAGISTERIUM] Empirical Probability of Saturation: {p_saturation * 100:.2f}%")
    print(f"[MAGISTERIUM] 95% HDI for Traffic Intensity (rho): {jnp.percentile(rho_samples, jnp.array([2.5, 97.5]))}")
    
    # Evaluate DreamLLM's Heuristic Bound
    # If the bound claims the queue is stable, but empirical saturation > 5%, trigger ANATHEMA.
    is_stable_according_to_bound = proposed_bound_func(rho_samples)
    
    if is_stable_according_to_bound and p_saturation > 0.05:
        print("\n[!] ANATHEMA: Proposed bound violated thermodynamic reality.")
        print("[!] Feedback to DreamLLM: False Positive on Stability. Mutate bound.")
        return False
    elif not is_stable_according_to_bound and p_saturation < 0.01:
        print("\n[!] ANATHEMA: Proposed bound is overly restrictive (False Negative).")
        print("[!] Feedback to DreamLLM: Mutate bound for higher capacity.")
        return False
    else:
        print("\n[+] APPROVED: Proposed bound aligns with heavy-tailed reality.")
        return True

# --- MOCK DREAMLLM HEURISTIC (Candidate Bound) ---
def dreamllm_candidate_bound(rho_samples):
    """
    This is the function DreamLLM iteratively rewrites.
    Example: LLM proposes that if the 90th percentile of rho is < 1.0, it's stable.
    """
    p90_rho = jnp.percentile(rho_samples, 90.0)
    return p90_rho < 1.0

if __name__ == "__main__":
    # SYNTHETIC DATA GENERATION: Heavy-Tailed Burst Traffic (The Adversarial Environment)
    true_lambda = 45.0
    k_true = 0.8  # k < 1 indicates heavy-tailed (high variance, "bursty" traffic/traffic jams)
    scale_true = 0.05
    c = 3
    
    key = jax.random.PRNGKey(42)
    key_arr, key_srv = jax.random.split(key)
    
    # Generate arrivals
    simulated_arrivals = jax.random.poisson(key_arr, true_lambda, shape=(200,))
    # Generate Weibull service times using inverse transform sampling
    U = jax.random.uniform(key_srv, shape=(2000,))
    simulated_service_times = scale_true * jnp.power(-jnp.log(U), 1.0 / k_true)
    
    print("=== SWEETBRIER: GENERAL QUEUE BOUND EVALUATION ===")
    print("Simulating Heavy-Tailed (Bursty) Traffic Dynamics...")
    
    # Run the Verifier
    success = run_magisterium_evaluation(
        simulated_arrivals, 
        simulated_service_times, 
        c, 
        dreamllm_candidate_bound
    )
