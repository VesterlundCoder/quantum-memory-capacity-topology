"""M1.B6: Analytical verification — closed form for single link and two-link chain.

Replace Table IV with closed-form verification:
  - Single link: A = 1-(1-p_eff)^n_max where p_eff = 1-(1-p_gen)^m
  - Two-link chain: A = 1-(1-p_eff^2)^n_max (swap always succeeds)
  - Compare against simulator

Also M1.B1: Absolute target variable — compare formula against A(m) in absolute terms.
"""
import sys, json
import numpy as np
from pathlib import Path
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wq_buffering.contention_sim import simulate_contention
from wq_buffering.comprehensive_eval import SwapConfig, make_decoherence_model, MEMORY_PROFILES, MemoryFamily
from wq_buffering.noise import NoiseType

print("=" * 80)
print("M1.B1+B6: Absolute target variable + analytical verification")
print("=" * 80)

# Parameters
P_GEN = 0.005
T_ATTEMPT = 1e-3
DEADLINE = 10.0
N_MAX = int(DEADLINE / T_ATTEMPT)  # = 10000
N_SEEDS = 10
N_REQUESTS = 1000

M_VALUES = [1, 2, 4, 8, 16, 32, 64]

print(f"p_gen={P_GEN}, t_attempt={T_ATTEMPT}, deadline={DEADLINE}")
print(f"n_max={N_MAX}, seeds={N_SEEDS}, requests={N_REQUESTS}")
print()

# === Analytical formulas ===
def single_link_A(m, p_gen=P_GEN, n_max=N_MAX):
    """Closed form for single link: A = 1-(1-p_eff)^n_max."""
    p_eff = 1 - (1 - p_gen) ** m
    return 1 - (1 - p_eff) ** n_max

def two_link_A(m, p_gen=P_GEN, n_max=N_MAX):
    """Closed form for two-link chain: A = 1-(1-p_eff^2)^n_max."""
    p_eff = 1 - (1 - p_gen) ** m
    p_path = p_eff ** 2
    return 1 - (1 - p_path) ** n_max

# === Run simulator ===
# Single link
G1 = nx.Graph()
G1.add_edge(0, 1)

# Two-link chain
G2 = nx.path_graph(3)

# Use Trapped Ion (long-lived, minimal retrieval effect)
profile = MEMORY_PROFILES[MemoryFamily.TRAPPED_ION]
dec = make_decoherence_model(NoiseType.DEPOLARIZING, 'medium', profile)

# Config with retrieval off and immediate storage (harmonized)
config = SwapConfig(
    p_gen=P_GEN, t_attempt=T_ATTEMPT,
    F_initial=0.99, deadline=DEADLINE, F_min=0.5,
    T_obs=200.0, warmup_fraction=0.1,
    retrieval_in_delivery=False,
    storage_time_model='immediate',
)

print(f"{'m':>4} {'A_sim_1link':>12} {'A_facit_1link':>14} {'ratio_1':>8} {'A_sim_2link':>12} {'A_facit_2link':>14} {'ratio_2':>8}")
print("-" * 80)

results = []
for m in M_VALUES:
    # Single link
    A_sim_1 = []
    for seed in range(N_SEEDS):
        r = simulate_contention(
            topology=G1, placement={0: m, 1: m},
            profile=profile, decoherence=dec,
            noise_type=NoiseType.DEPOLARIZING, config=config,
            arrival_rate=1.0, n_requests=N_REQUESTS, random_seed=seed,
        )
        A_sim_1.append(r['A_req'])
    A_sim_1_mean = np.mean(A_sim_1)
    A_facit_1 = single_link_A(m)
    ratio_1 = A_sim_1_mean / max(A_facit_1, 1e-10)
    
    # Two-link chain
    A_sim_2 = []
    for seed in range(N_SEEDS):
        r = simulate_contention(
            topology=G2, placement={v: m for v in G2.nodes()},
            profile=profile, decoherence=dec,
            noise_type=NoiseType.DEPOLARIZING, config=config,
            arrival_rate=1.0, n_requests=N_REQUESTS, random_seed=seed,
        )
        A_sim_2.append(r['A_req'])
    A_sim_2_mean = np.mean(A_sim_2)
    A_facit_2 = two_link_A(m)
    ratio_2 = A_sim_2_mean / max(A_facit_2, 1e-10)
    
    results.append({
        'm': m,
        'A_sim_1link': A_sim_1_mean,
        'A_facit_1link': A_facit_1,
        'ratio_1': ratio_1,
        'A_sim_2link': A_sim_2_mean,
        'A_facit_2link': A_facit_2,
        'ratio_2': ratio_2,
    })
    
    print(f"{m:4d} {A_sim_1_mean:12.4f} {A_facit_1:14.4f} {ratio_1:8.4f} {A_sim_2_mean:12.4f} {A_facit_2:14.4f} {ratio_2:8.4f}")

# === Clipping at 64 ===
print("\n" + "=" * 80)
print("M1.B1: Clipping at m=64")
print("=" * 80)
print(f"Max m tested: {max(M_VALUES)}")
print(f"A at m=64 (single link): {results[-1]['A_sim_1link']:.4f} (facit: {results[-1]['A_facit_1link']:.4f})")
print(f"A at m=64 (two-link):     {results[-1]['A_sim_2link']:.4f} (facit: {results[-1]['A_facit_2link']:.4f})")
print(f"Clipping: A is clipped at 1.0 (saturation). For m>64, A would remain at ~1.0.")

# === Save ===
output = {
    "p_gen": P_GEN,
    "t_attempt": T_ATTEMPT,
    "deadline": DEADLINE,
    "n_max": N_MAX,
    "n_seeds": N_SEEDS,
    "n_requests": N_REQUESTS,
    "m_values": M_VALUES,
    "results": results,
    "formulas": {
        "single_link": "A = 1-(1-p_eff)^n_max, p_eff = 1-(1-p_gen)^m",
        "two_link": "A = 1-(1-p_eff^2)^n_max, p_eff = 1-(1-p_gen)^m",
    },
}

output_path = Path(__file__).parent / "analytical_verification.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
