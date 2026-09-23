import os
import re
import jax
import jax.numpy as jnp
from bayesian_queue_verifier import run_empirical_verification

# ==============================================================================
# HEURISTIC DISCOVERY LOOP
# Purpose: The cybernetic loop that forces an LLM to evolve capacity bounds 
#          until the deterministic verifier cannot falsify them.
# ==============================================================================

def query_llm_agent(prompt: str) -> str:
    print("\n[LLM_Agent] Generating candidate mathematical bound...")
    # NOTE FOR VIVIAN: Plug your API call to Gemini/Claude here.
    
    mock_llm_response = '''
def llm_candidate_bound(rho_samples):
    import jax.numpy as jnp
    p95_rho = jnp.percentile(rho_samples, 95.0)
    tail_volatility = jnp.var(rho_samples)
    return (p95_rho + (0.1 * tail_volatility)) < 1.0
'''
    return mock_llm_response

def extract_python_code(llm_response: str) -> str:
    match = re.search(r'`python\n(.*?)`', llm_response, re.DOTALL)
    return match.group(1) if match else llm_response

def start_discovery_loop():
    true_lambda = 45.0
    k_true = 0.8
    scale_true = 0.05
    c = 3
    key = jax.random.PRNGKey(42)
    key_arr, key_srv = jax.random.split(key)
    simulated_arrivals = jax.random.poisson(key_arr, true_lambda, shape=(500,))
    U = jax.random.uniform(key_srv, shape=(5000,))
    simulated_service_times = scale_true * jnp.power(-jnp.log(U), 1.0 / k_true)
    
    context = """
    You are a theoretical computer scientist. We are trying to find the capacity bound 
    for a non-memoryless M/G/c queue with feedback. 
    Write a Python function named llm_candidate_bound(rho_samples) that takes 
    a JAX array of posterior traffic intensity (rho) samples and returns a boolean True 
    if the queue is strictly stable, and False otherwise. 
    Keep it strictly to pure math/JAX operations. Return ONLY code.
    """
    
    iteration = 1
    max_iterations = 50
    
    while iteration <= max_iterations:
        print(f"\n{'='*50}\nDISCOVERY ITERATION: {iteration}\n{'='*50}")
        
        llm_output = query_llm_agent(context)
        python_code = extract_python_code(llm_output)
        
        namespace = {}
        try:
            exec(python_code, namespace)
            candidate_func = namespace['llm_candidate_bound']
        except Exception as e:
            print(f"[!] Compilation Error in LLM output: {e}")
            context = f"Your last code failed to compile with error: {e}. Fix it and return only code."
            iteration += 1
            continue
            
        print("[System] Executing candidate bound in the Empirical Verifier...")
        success = run_empirical_verification(simulated_arrivals, simulated_service_times, c, candidate_func)
        
        if success:
            print("\n[***] EMPIRICAL CONVERGENCE ACHIEVED [***]")
            print("The LLM generated a capacity bound that the Verifier could not falsify.")
            print("Verified Code Output:")
            print(python_code)
            break
        else:
            print("\n[!] Bound Failed. Sending rejection trace back to LLM for mutation.")
            context = f"""
            Your previous function failed the empirical simulation. 
            It either returned True when the actual queue saturated >5% of the time, 
            or False when the queue was perfectly stable. 
            Adjust the tail sensitivity in your calculation. 
            Here was your failed code:\n{python_code}\n\nReturn the mutated function.
            """
        
        iteration += 1

if __name__ == "__main__":
    start_discovery_loop()
