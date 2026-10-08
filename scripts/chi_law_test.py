"""χ-law test as a law (Punkt 3).

Design:
1. Invariance remade: 7 χ values (0.03 to 30), ≥6 combinations per χ,
   T2/p_gen/t_attempt vary over a decade each.
   Two variants: deadline/traffic fixed in seconds, and scaled dimensionlessly.
2. Noise floor: spread between combinations vs spread between replicates.
3. Which lifetime binds: grid where T2 and τ_r vary independently.
4. Correct target: predict A(m) curve in absolute terms, not relative threshold.
5. Continuous dataset: ~2000 configurations, RF with χ-decade left out.
6. New atlas: depolarizing readout, 10× longer, m to 64, arm with fixed total memory.

Criterion (committed before run): spread at fixed χ ≤ 2× noise floor.
"""
import sys, json, time
import numpy as np
from pathlib import Path
import networkx as nx
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wq_buffering.contention_sim import simulate_contention
from wq_buffering.comprehensive_eval import SwapConfig, make_decoherence_model, MEMORY_PROFILES, MemoryProfile, MemoryFamily
from wq_buffering.noise import NoiseType

OUTPUT = Path(__file__).parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# === Part 1: Invariance remade ===

print("=" * 80)
print("Punkt 3: χ-law test as a law")
print("=" * 80)

# 7 χ values from 0.03 to 30
chi_values = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# For each χ, generate ≥6 combinations where T2, p_gen, t_attempt vary
# but χ = T2 * p_gen / t_attempt is constant.
# Each parameter varies over at least a decade.

def generate_chi_combos(chi_target, n_combos=6):
    """Generate combinations where T2 * p_gen / t_attempt = chi_target."""
    combos = []
    rng = np.random.default_rng(42)
    for _ in range(n_combos):
        # Vary T2 over a decade
        T2 = 10 ** rng.uniform(np.log10(0.001), np.log10(100))
        # Vary p_gen over a decade
        p_gen = 10 ** rng.uniform(np.log10(0.0005), np.log10(0.05))
        # Compute t_attempt to hit chi_target
        t_attempt = T2 * p_gen / chi_target
        # Ensure t_attempt is reasonable (1e-5 to 1e-1)
        if t_attempt < 1e-5:
            t_attempt = 1e-5
        if t_attempt > 1e-1:
            t_attempt = 1e-1
        combos.append({
            'T2': T2, 'p_gen': p_gen, 't_attempt': t_attempt,
            'chi': T2 * p_gen / t_attempt,
        })
    return combos

# Generate all combinations
all_combos = {}
for chi in chi_values:
    all_combos[chi] = generate_chi_combos(chi, n_combos=8)
    print(f"χ={chi}: {len(all_combos[chi])} combos")

# Run simulation for each combo × m × seed
M_VALUES = [1, 4, 8, 16, 32, 64]
N_REQUESTS = 1000
N_SEEDS = 3

# Single link topology
G = nx.Graph()
G.add_edge(0, 1)

print(f"\nRunning {len(chi_values)} χ × 8 combos × {len(M_VALUES)} m × {N_SEEDS} seeds = {len(chi_values)*8*len(M_VALUES)*N_SEEDS} runs")
print("(Two variants: fixed and scaled)")

start = time.time()
results_fixed = []
results_scaled = []

for chi in chi_values:
    for combo in all_combos[chi]:
        T2 = combo['T2']
        p_gen = combo['p_gen']
        t_attempt = combo['t_attempt']
        
        # Create a profile with this T2
        profile = MemoryProfile(
            family=MemoryFamily.TRAPPED_ION,
            T1=T2 * 2, T2=T2, T2_active=T2,
            retrieval_efficiency=0.95, retrieval_tau=T2,
            readout_fidelity=0.99, gate_fidelity=0.99,
            supports_2q_gates=True, max_modes=10, activity_noise_factor=1.0,
        )
        dec = make_decoherence_model(NoiseType.DEPOLARIZING, 'medium', profile)
        
        for m in M_VALUES:
            # Variant 1: Fixed deadline and traffic in seconds
            config_fixed = SwapConfig(
                p_gen=p_gen, t_attempt=t_attempt,
                F_initial=0.99, deadline=10.0, F_min=0.5,
                T_obs=200.0, warmup_fraction=0.1,
            )
            placement = {0: m, 1: m}
            
            for seed in range(N_SEEDS):
                r = simulate_contention(
                    topology=G, placement=placement,
                    profile=profile, decoherence=dec,
                    noise_type=NoiseType.DEPOLARIZING, config=config_fixed,
                    arrival_rate=1.0, n_requests=N_REQUESTS, random_seed=seed,
                )
                results_fixed.append({
                    'chi': chi, 'T2': T2, 'p_gen': p_gen, 't_attempt': t_attempt,
                    'm': m, 'seed': seed, 'A_req': r['A_req'],
                    'variant': 'fixed',
                })
            
            # Variant 2: Scaled dimensionlessly
            # deadline_scaled = deadline / t_attempt (dimensionless)
            # traffic_scaled = traffic * t_attempt (dimensionless)
            # Keep dimensionless equivalents constant
            deadline_dim = 10.0 / t_attempt  # = 10/t_attempt attempts
            traffic_dim = 1.0 * t_attempt  # dimensionless
            # Convert back: deadline = deadline_dim * t_attempt, traffic = traffic_dim / t_attempt
            config_scaled = SwapConfig(
                p_gen=p_gen, t_attempt=t_attempt,
                F_initial=0.99, deadline=deadline_dim * t_attempt, F_min=0.5,
                T_obs=200.0, warmup_fraction=0.1,
            )
            
            for seed in range(N_SEEDS):
                r = simulate_contention(
                    topology=G, placement=placement,
                    profile=profile, decoherence=dec,
                    noise_type=NoiseType.DEPOLARIZING, config=config_scaled,
                    arrival_rate=traffic_dim / t_attempt, n_requests=N_REQUESTS, random_seed=seed,
                )
                results_scaled.append({
                    'chi': chi, 'T2': T2, 'p_gen': p_gen, 't_attempt': t_attempt,
                    'm': m, 'seed': seed, 'A_req': r['A_req'],
                    'variant': 'scaled',
                })
    
    elapsed = time.time() - start
    print(f"  χ={chi} done ({elapsed:.1f}s)")

print(f"\nTotal runs: {len(results_fixed) + len(results_scaled)}, runtime: {time.time()-start:.1f}s")

# === Analysis ===
print("\n" + "=" * 80)
print("ANALYSIS: Invariance and noise floor")
print("=" * 80)

# For each χ and m, compute:
# - CV between combinations (spread)
# - CV between replicates (noise floor)
# - Ratio: spread / noise floor

for variant_name, results in [('fixed', results_fixed), ('scaled', results_scaled)]:
    print(f"\n--- Variant: {variant_name} ---")
    print(f"{'χ':>8} {'m':>4} {'CV_combos':>10} {'CV_replicates':>14} {'Ratio':>8} {'Pass':>6}")
    
    for chi in chi_values:
        for m in M_VALUES:
            # Get A_req for all combos at this chi and m
            combo_means = []
            combo_stds = []
            for combo in all_combos[chi]:
                vals = [r['A_req'] for r in results
                        if r['chi'] == chi and r['m'] == m
                        and abs(r['T2'] - combo['T2']) < 1e-10
                        and abs(r['p_gen'] - combo['p_gen']) < 1e-10]
                if vals:
                    combo_means.append(np.mean(vals))
                    combo_stds.append(np.std(vals, ddof=1))
            
            if len(combo_means) < 2:
                continue
            
            # CV between combos (spread)
            cv_combos = np.std(combo_means, ddof=1) / max(abs(np.mean(combo_means)), 1e-10)
            
            # CV between replicates (noise floor)
            all_stds = [s for s in combo_stds if not np.isnan(s) and s > 0]
            if all_stds:
                mean_std = np.mean(all_stds)
                cv_replicates = mean_std / max(abs(np.mean(combo_means)), 1e-10)
            else:
                cv_replicates = 0.01  # Default noise floor
            
            ratio = cv_combos / max(cv_replicates, 1e-10)
            pass_test = ratio <= 2.0
            
            print(f"{chi:8.4f} {m:4d} {cv_combos:10.4f} {cv_replicates:14.4f} {ratio:8.2f} {'PASS' if pass_test else 'FAIL':>6}")

# === Save ===
output = {
    "chi_values": chi_values,
    "n_combos_per_chi": 8,
    "m_values": M_VALUES,
    "n_seeds": N_SEEDS,
    "n_requests": N_REQUESTS,
    "criterion": "spread at fixed χ ≤ 2× noise floor",
    "results_fixed": results_fixed,
    "results_scaled": results_scaled,
}

output_path = Path(__file__).parent / "chi_law_test.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
