"""χ-law invariance test — extended variants (M1.A1, A2, A4, A6).

A1: Invariance without retrieval (retrieval_in_delivery=False)
A2: Invariance with τ_r scaled (τ_r/T₂ constant across combos)
A4: Deadline and traffic scaled (dimensionless equivalents constant)
A6: 10 seeds per cell (stable noise floor)

Four variants:
  V1: Original (retrieval on, τ_r=T2, fixed deadline/traffic)
  V2: Retrieval off (retrieval_in_delivery=False)
  V3: τ_r scaled (τ_r/T₂ = constant ratio across combos)
  V4: Scaled dimensionless (deadline/traffic scaled with t_attempt)

Criterion (committed before run): spread at fixed χ ≤ 2× noise floor.
Report: pass/fail + magnitude of deviation in each cell.
"""
import sys, json, time
import numpy as np
from pathlib import Path
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wq_buffering.contention_sim import simulate_contention
from wq_buffering.comprehensive_eval import SwapConfig, make_decoherence_model, MemoryProfile, MemoryFamily
from wq_buffering.noise import NoiseType

OUTPUT = Path(__file__).parent

# === Configuration ===
chi_values = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
M_VALUES = [1, 4, 8, 16, 32, 64]
N_REQUESTS = 1000
N_SEEDS = 10  # A6: at least 10 per cell

# Single link topology
G = nx.Graph()
G.add_edge(0, 1)

# === Generate χ-preserving combinations ===
def generate_chi_combos(chi_target, n_combos=8, tau_r_ratio=None):
    """Generate combinations where T2 * p_gen / t_attempt = chi_target.
    
    If tau_r_ratio is given, set τ_r = tau_r_ratio * T2 (constant ratio).
    Otherwise τ_r = T2 (original).
    """
    combos = []
    rng = np.random.default_rng(42 + int(chi_target * 1000))
    for _ in range(n_combos):
        T2 = 10 ** rng.uniform(np.log10(0.001), np.log10(100))
        p_gen = 10 ** rng.uniform(np.log10(0.0005), np.log10(0.05))
        t_attempt = T2 * p_gen / chi_target
        if t_attempt < 1e-5:
            t_attempt = 1e-5
        if t_attempt > 1e-1:
            t_attempt = 1e-1
        
        if tau_r_ratio is not None:
            tau_r = tau_r_ratio * T2
        else:
            tau_r = T2  # Original: τ_r = T2
        
        combos.append({
            'T2': T2, 'p_gen': p_gen, 't_attempt': t_attempt,
            'chi': T2 * p_gen / t_attempt,
            'tau_r': tau_r,
        })
    return combos

# === Run all four variants ===
print("=" * 90)
print("χ-law invariance test — 4 variants × 10 seeds")
print("=" * 90)
print(f"χ values: {chi_values}")
print(f"m values: {M_VALUES}")
print(f"Seeds: {N_SEEDS}")
print(f"Variants: V1=original, V2=retrieval_off, V3=tau_r_scaled, V4=scaled_dimless")
print()

# Fixed τ_r/T₂ ratio for V3
TAU_R_RATIO = 0.1  # τ_r = 0.1 * T2 (constant ratio across all combos)

all_results = {}

for variant_name, tau_r_ratio, retrieval_off, scale_dimless in [
    ("V1_original", None, False, False),
    ("V2_retrieval_off", None, True, False),
    ("V3_tau_r_scaled", TAU_R_RATIO, False, False),
    ("V4_scaled_dimless", None, False, True),
]:
    print(f"\n--- {variant_name} ---")
    start = time.time()
    results = []
    
    for chi in chi_values:
        combos = generate_chi_combos(chi, n_combos=8, tau_r_ratio=tau_r_ratio)
        
        for combo in combos:
            T2 = combo['T2']
            p_gen = combo['p_gen']
            t_attempt = combo['t_attempt']
            tau_r = combo['tau_r']
            
            profile = MemoryProfile(
                family=MemoryFamily.TRAPPED_ION,
                T1=T2 * 2, T2=T2, T2_active=T2,
                retrieval_efficiency=0.95, retrieval_tau=tau_r,
                readout_fidelity=0.99, gate_fidelity=0.99,
                supports_2q_gates=True, max_modes=10, activity_noise_factor=1.0,
            )
            dec = make_decoherence_model(NoiseType.DEPOLARIZING, 'medium', profile)
            
            for m in M_VALUES:
                if scale_dimless:
                    # V4: scale deadline and traffic with t_attempt
                    # Keep dimensionless deadline = 10/t_attempt_ref constant
                    t_ref = 1e-3  # reference t_attempt
                    deadline = 10.0 * (t_attempt / t_ref)
                    traffic = 1.0 * (t_ref / t_attempt)
                else:
                    deadline = 10.0
                    traffic = 1.0
                
                config = SwapConfig(
                    p_gen=p_gen, t_attempt=t_attempt,
                    F_initial=0.99, deadline=deadline, F_min=0.5,
                    T_obs=200.0, warmup_fraction=0.1,
                    retrieval_in_delivery=not retrieval_off,
                )
                placement = {0: m, 1: m}
                
                for seed in range(N_SEEDS):
                    r = simulate_contention(
                        topology=G, placement=placement,
                        profile=profile, decoherence=dec,
                        noise_type=NoiseType.DEPOLARIZING, config=config,
                        arrival_rate=traffic, n_requests=N_REQUESTS, random_seed=seed,
                    )
                    results.append({
                        'chi': chi, 'T2': T2, 'p_gen': p_gen,
                        't_attempt': t_attempt, 'tau_r': tau_r,
                        'm': m, 'seed': seed, 'A_req': r['A_req'],
                        'variant': variant_name,
                    })
        
        elapsed = time.time() - start
        print(f"  χ={chi} done ({elapsed:.1f}s)")
    
    all_results[variant_name] = results
    print(f"  Total: {len(results)} runs, {time.time()-start:.1f}s")

# === Analysis ===
print("\n" + "=" * 90)
print("ANALYSIS: pass/fail + deviation magnitude")
print("=" * 90)

def analyze_variant(results, variant_name):
    """Analyze invariance for one variant."""
    print(f"\n--- {variant_name} ---")
    print(f"{'χ':>8} {'m':>4} {'CV_combos':>10} {'CV_repl':>10} {'Ratio':>8} {'Max_dev%':>10} {'Pass':>6}")
    print("-" * 70)
    
    table = []
    for chi in chi_values:
        for m in M_VALUES:
            combo_means = []
            combo_stds = []
            for T2 in set(r['T2'] for r in results if abs(r['chi'] - chi) < 1e-6):
                vals = [r['A_req'] for r in results
                        if abs(r['chi'] - chi) < 1e-6
                        and abs(r['T2'] - T2) < 1e-10
                        and r['m'] == m]
                if vals:
                    combo_means.append(np.mean(vals))
                    combo_stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            
            if len(combo_means) < 2:
                continue
            
            cv_combos = np.std(combo_means, ddof=1) / max(abs(np.mean(combo_means)), 1e-10)
            all_stds = [s for s in combo_stds if not np.isnan(s) and s > 0]
            cv_repl = np.mean(all_stds) / max(abs(np.mean(combo_means)), 1e-10) if all_stds else 0.01
            ratio = cv_combos / max(cv_repl, 1e-10)
            max_dev = np.max(np.abs(combo_means - np.mean(combo_means))) / max(abs(np.mean(combo_means)), 1e-10)
            pass_test = ratio <= 2.0
            
            table.append({
                'chi': chi, 'm': m,
                'cv_combos': cv_combos, 'cv_repl': cv_repl,
                'ratio': ratio, 'max_dev': max_dev,
                'pass': pass_test,
            })
            
            print(f"{chi:8.4f} {m:4d} {cv_combos:10.4f} {cv_repl:10.4f} {ratio:8.2f} {max_dev*100:10.1f} {'PASS' if pass_test else 'FAIL':>6}")
    
    return table

tables = {}
for variant_name, results in all_results.items():
    tables[variant_name] = analyze_variant(results, variant_name)

# === Summary: which variants pass where? ===
print("\n" + "=" * 90)
print("SUMMARY: pass/fail across variants")
print("=" * 90)
print(f"{'χ':>8} {'m':>4} {'V1':>6} {'V2':>6} {'V3':>6} {'V4':>6}")
print("-" * 40)
for chi in chi_values:
    for m in M_VALUES:
        row = []
        for v in ['V1_original', 'V2_retrieval_off', 'V3_tau_r_scaled', 'V4_scaled_dimless']:
            entry = [t for t in tables[v] if t['chi'] == chi and t['m'] == m]
            if entry:
                row.append('PASS' if entry[0]['pass'] else 'FAIL')
            else:
                row.append('---')
        print(f"{chi:8.4f} {m:4d} {row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>6}")

# === Key question A1: does retrieval off improve invariance? ===
print("\n" + "=" * 90)
print("A1: Does retrieval off improve invariance?")
print("=" * 90)
v1_pass = sum(1 for t in tables['V1_original'] if t['pass'])
v2_pass = sum(1 for t in tables['V2_retrieval_off'] if t['pass'])
print(f"V1 (original):     {v1_pass}/{len(tables['V1_original'])} pass")
print(f"V2 (retrieval off): {v2_pass}/{len(tables['V2_retrieval_off'])} pass")
print(f"Improvement: {v2_pass - v1_pass} cells")

# === Key question A2: does τ_r scaling restore low-χ? ===
print("\n" + "=" * 90)
print("A2: Does τ_r/T₂ constant restore low-χ regime?")
print("=" * 90)
for chi in [0.03, 0.1, 0.3]:
    v1 = [t for t in tables['V1_original'] if t['chi'] == chi]
    v3 = [t for t in tables['V3_tau_r_scaled'] if t['chi'] == chi]
    v1_pass = sum(1 for t in v1 if t['pass'])
    v3_pass = sum(1 for t in v3 if t['pass'])
    print(f"χ={chi}: V1 {v1_pass}/{len(v1)} pass → V3 {v3_pass}/{len(v3)} pass")

# === Save ===
output = {
    "n_seeds": N_SEEDS,
    "n_requests": N_REQUESTS,
    "chi_values": chi_values,
    "m_values": M_VALUES,
    "criterion": "spread at fixed χ ≤ 2× noise floor",
    "variants": {
        "V1_original": "retrieval on, τ_r=T2, fixed deadline/traffic",
        "V2_retrieval_off": "retrieval_in_delivery=False",
        "V3_tau_r_scaled": f"τ_r/T₂={TAU_R_RATIO} constant",
        "V4_scaled_dimless": "deadline/traffic scaled with t_attempt",
    },
    "tables": tables,
    "results": {k: v for k, v in all_results.items()},
}

output_path = OUTPUT / "chi_invariance_extended.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
