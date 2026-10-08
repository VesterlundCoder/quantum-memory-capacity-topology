"""Harder closure test + full 2^4 factorial with retrieval_in_delivery.

Closure test uses parameters where A_facit < 1.0 (short deadline, low p_gen).
Full factorial: 4 switches × 2 levels = 16 variants.
"""
import sys, json, time, itertools
import numpy as np
from pathlib import Path
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wq_buffering.contention_sim import simulate_contention
from wq_buffering.comprehensive_eval import SwapConfig, make_decoherence_model, MEMORY_PROFILES, MemoryFamily
from wq_buffering.noise import NoiseType

OUTPUT = Path(__file__).parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# === Harder closure test ===
print("=" * 80)
print("Harder closure test (A_facit < 1.0)")
print("=" * 80)

G = nx.Graph()
G.add_edge(0, 1)

# Use parameters where A_facit is nontrivial
p_gen = 0.001  # Low generation probability
t_attempt = 1e-3
deadline = 1.0  # Short deadline
n_max = int(deadline / t_attempt)  # = 1000

print(f"p_gen={p_gen}, t_attempt={t_attempt}, deadline={deadline}, n_max={n_max}")
print()

all_pass = True
closure_results = []

for fam in MemoryFamily:
    profile = MEMORY_PROFILES[fam]
    dec = make_decoherence_model(NoiseType.DEPOLARIZING, 'medium', profile)
    
    for m in [1, 4, 16]:
        # Harmonized mode: immediate, no retrieval, queue, parallel
        config = SwapConfig(
            p_gen=p_gen, t_attempt=t_attempt,
            F_initial=0.99, deadline=deadline, F_min=0.0,
            T_obs=200.0, warmup_fraction=0.1,
            retrieval_in_delivery=False,
            storage_time_model='immediate',
            blocking_policy='queue',
            generation_mode='parallel',
        )
        placement = {0: m, 1: m}
        
        A_vals = []
        for seed in range(10):
            r = simulate_contention(
                topology=G, placement=placement,
                profile=profile, decoherence=dec,
                noise_type=NoiseType.DEPOLARIZING, config=config,
                arrival_rate=0.5, n_requests=2000, random_seed=seed,
            )
            A_vals.append(r['A_req'])
        
        A_sim = np.mean(A_vals)
        A_std = np.std(A_vals, ddof=1)
        se = A_std / np.sqrt(10)
        
        # Facit: A = 1 - (1-p_eff)^n_max
        p_eff = 1 - (1 - p_gen) ** m
        A_facit = 1 - (1 - p_eff) ** n_max
        
        within_3se = abs(A_sim - A_facit) <= 3 * se
        ratio = A_sim / max(A_facit, 1e-10)
        
        pass_test = within_3se
        if not pass_test:
            all_pass = False
        
        closure_results.append({
            'family': fam.name, 'm': m,
            'A_sim': A_sim, 'A_facit': A_facit,
            'A_std': A_std, 'se': se,
            'ratio': ratio, 'pass': pass_test,
        })
        
        print(f"  {fam.name:>20} m={m:3d} A_sim={A_sim:.4f}±{se:.4f} A_facit={A_facit:.4f} ratio={ratio:.4f} {'PASS' if pass_test else 'FAIL'}")

print(f"\nClosure test: {'ALL PASS' if all_pass else 'SOME FAIL'}")

# === Full 2^4 factorial ===
print()
print("=" * 80)
print("Full 2^4 factorial: 4 switches × 2 levels = 16 variants")
print("=" * 80)

SWITCHES = {
    'retrieval_in_delivery': [True, False],
    'storage_time_model': ['midpoint', 'immediate'],
    'blocking_policy': ['queue', 'reject'],
    'generation_mode': ['parallel', 'serial'],
}

switch_names = list(SWITCHES.keys())
switch_levels = list(SWITCHES.values())
variants = list(itertools.product(*switch_levels))
print(f"Variants: {len(variants)}")

# 3 families spanning χ
FAMILIES = [MemoryFamily.TRAPPED_ION, MemoryFamily.NEUTRAL_ATOM, MemoryFamily.AFC]
N = 5
M = 2
TRAFFIC = 50.0
N_REQUESTS = 1000
N_SEEDS = 5

G = nx.path_graph(N)

start = time.time()
results = []

for fam in FAMILIES:
    profile = MEMORY_PROFILES[fam]
    dec = make_decoherence_model(NoiseType.DEPOLARIZING, 'medium', profile)
    
    for var_idx, variant in enumerate(variants):
        config = SwapConfig(
            p_gen=0.005, t_attempt=1e-3,
            F_initial=0.99, deadline=10.0, F_min=0.5,
            T_obs=200.0, warmup_fraction=0.1,
            retrieval_in_delivery=variant[0],
            storage_time_model=variant[1],
            blocking_policy=variant[2],
            generation_mode=variant[3],
        )
        placement = {v: M for v in G.nodes()}
        
        for seed in range(N_SEEDS):
            r = simulate_contention(
                topology=G, placement=placement,
                profile=profile, decoherence=dec,
                noise_type=NoiseType.DEPOLARIZING, config=config,
                arrival_rate=TRAFFIC, n_requests=N_REQUESTS, random_seed=seed,
            )
            results.append({
                'family': fam.name,
                'retrieval_in_delivery': variant[0],
                'storage_time_model': variant[1],
                'blocking_policy': variant[2],
                'generation_mode': variant[3],
                'variant_idx': var_idx,
                'seed': seed,
                'A_req': r['A_req'],
            })
    
    print(f"  {fam.name} done ({time.time()-start:.1f}s)")

print(f"\nTotal runs: {len(results)}, runtime: {time.time()-start:.1f}s")

# === Variance decomposition ===
print()
print("=" * 80)
print("VARIANCE DECOMPOSITION (4 switches)")
print("=" * 80)

def variance_decomposition(results, family):
    fam_results = [r for r in results if r['family'] == family]
    if not fam_results:
        return None
    
    all_A = np.array([r['A_req'] for r in fam_results])
    grand_mean = np.mean(all_A)
    total_var = np.var(all_A, ddof=1)
    
    if total_var < 1e-12:
        return {'total_var': 0, 'main_effects': {}, 'interactions': {}}
    
    main_effects = {}
    for switch in switch_names:
        levels = SWITCHES[switch]
        level_means = {}
        for level in levels:
            vals = [r['A_req'] for r in fam_results if r[switch] == level]
            if vals:
                level_means[level] = np.mean(vals)
        
        ss = 0
        for level in levels:
            n_level = sum(1 for r in fam_results if r[switch] == level)
            if n_level > 0:
                ss += n_level * (level_means[level] - grand_mean) ** 2
        
        main_effects[switch] = {
            'ss': ss,
            'var_explained': ss / (total_var * len(all_A)) if total_var > 0 else 0,
            'level_means': level_means,
        }
    
    # Two-way interactions
    interactions = {}
    for i, s1 in enumerate(switch_names):
        for s2 in switch_names[i+1:]:
            cells = {}
            for l1 in SWITCHES[s1]:
                for l2 in SWITCHES[s2]:
                    vals = [r['A_req'] for r in fam_results
                            if r[s1] == l1 and r[s2] == l2]
                    if vals:
                        cells[(l1, l2)] = np.mean(vals)
            
            ss_cells = 0
            for (l1, l2), cell_mean in cells.items():
                n_cell = sum(1 for r in fam_results
                             if r[s1] == l1 and r[s2] == l2)
                if n_cell > 0:
                    expected = main_effects[s1]['level_means'][l1] + \
                               main_effects[s2]['level_means'][l2] - grand_mean
                    ss_cells += n_cell * (cell_mean - expected) ** 2
            
            interactions[f'{s1} x {s2}'] = {
                'ss': ss_cells,
                'var_explained': ss_cells / (total_var * len(all_A)) if total_var > 0 else 0,
            }
    
    return {
        'total_var': total_var,
        'grand_mean': grand_mean,
        'main_effects': main_effects,
        'interactions': interactions,
    }

decomposition = {}
for fam in FAMILIES:
    dec = variance_decomposition(results, fam.name)
    if dec:
        decomposition[fam.name] = dec
        print(f"\n  {fam.name} (mean A_req={dec['grand_mean']:.4f}, var={dec['total_var']:.6f}):")
        print(f"    Main effects:")
        for switch, info in dec['main_effects'].items():
            print(f"      {switch:30s}: {info['var_explained']*100:.1f}%")
        print(f"    Two-way interactions:")
        for name, info in dec['interactions'].items():
            if info['var_explained'] > 0.01:
                print(f"      {name:50s}: {info['var_explained']*100:.1f}%")

# === Cluster bootstrap ===
print()
print("=" * 80)
print("CLUSTER BOOTSTRAP (1000 resamples)")
print("=" * 80)

def cluster_bootstrap(results, family, n_boot=1000):
    fam_results = [r for r in results if r['family'] == family]
    if not fam_results:
        return None
    
    seeds = list(set(r['seed'] for r in fam_results))
    rng = np.random.default_rng(42)
    
    boot_effects = {switch: [] for switch in switch_names}
    
    for _ in range(n_boot):
        boot_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_results = []
        for s in boot_seeds:
            boot_results.extend([r for r in fam_results if r['seed'] == s])
        
        all_A = np.array([r['A_req'] for r in boot_results])
        total_var = np.var(all_A, ddof=1)
        if total_var < 1e-12:
            continue
        grand_mean = np.mean(all_A)
        
        for switch in switch_names:
            levels = SWITCHES[switch]
            ss = 0
            for level in levels:
                vals = [r['A_req'] for r in boot_results if r[switch] == level]
                if vals:
                    level_mean = np.mean(vals)
                    n_level = len(vals)
                    ss += n_level * (level_mean - grand_mean) ** 2
            boot_effects[switch].append(ss / (total_var * len(all_A)))
    
    ci = {}
    for switch in switch_names:
        if boot_effects[switch]:
            ci[switch] = {
                'mean': np.mean(boot_effects[switch]),
                'ci_low': np.percentile(boot_effects[switch], 2.5),
                'ci_high': np.percentile(boot_effects[switch], 97.5),
            }
    return ci

bootstrap_results = {}
for fam in FAMILIES:
    ci = cluster_bootstrap(results, fam.name)
    if ci:
        bootstrap_results[fam.name] = ci
        print(f"\n  {fam.name}:")
        for switch, info in ci.items():
            print(f"    {switch:30s}: {info['mean']*100:.1f}% [{info['ci_low']*100:.1f}, {info['ci_high']*100:.1f}]")

# === Save ===
output = {
    "closure_test": closure_results,
    "closure_all_pass": all_pass,
    "n_variants": len(variants),
    "n_families": len(FAMILIES),
    "n_seeds": N_SEEDS,
    "n_requests": N_REQUESTS,
    "switches": SWITCHES,
    "results": results,
    "variance_decomposition": decomposition,
    "bootstrap_ci": bootstrap_results,
}

output_path = Path(__file__).parent / "closure_and_factorial4.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
