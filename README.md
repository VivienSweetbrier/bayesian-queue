# Bayesian Queueing System

An empirical verification framework for discovering and evaluating capacity bounds in non-memoryless (M/G/c) queue channels with feedback. 

## Overview
Traditional queueing theory relies on memoryless (Markovian) assumptions (e.g., M/M/c queues) to maintain algebraic tractability. However, real-world information flows—such as network traffic, supply chains, and datacenter workloads—exhibit bursty, heavy-tailed service distributions. Deriving exact theoretical capacity bounds for these generalized queues with feedback remains an open problem in computer science and operations research.

This repository implements an automated discovery loop that pairs a generative heuristic search with a rigorous Bayesian Monte Carlo verifier to empirically falsify candidate mathematical bounds against simulated heavy-tailed traffic.

## Architecture

The system is divided into two primary components:

### 1. Empirical Verifier (`bayesian_queue_verifier.py`)
A probabilistic simulation engine built with JAX and NumPyro.
- **Data Generation**: Generates chaotic, heavy-tailed traffic flows using Weibull distributions to simulate realistic, non-memoryless service times.
- **Bayesian Inference**: Executes Hamiltonian Monte Carlo (NUTS) to estimate the posterior distribution of the queue's traffic intensity ($\rho$).
- **Verification Logic**: Evaluates candidate mathematical bounds against the empirical posterior. If a proposed bound predicts stability but the simulation empirically exceeds a predefined saturation threshold (e.g., >5% saturation rate), the bound is rejected.

### 2. Heuristic Discovery Loop (`heuristic_discovery_loop.py`)
An orchestration script designed to automate the search for valid theoretical bounds via iterative mutation.
- Iteratively queries an external generative model to propose candidate capacity bounds as executable JAX/Python functions.
- Dynamically compiles and evaluates these functions against the Empirical Verifier.
- Feeds failure traces back to the generator for continuous mutation until empirical convergence is achieved across the tested distributions.

## Dependencies
- `jax`
- `jaxlib`
- `numpyro`
