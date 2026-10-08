# Cover Letter — Paper A for Quantum Reports

**To:** The Editor, Quantum Reports (MDPI)

**Date:** 6 October 2026

**Manuscript title:** Memory Physics, Finite Capacity, and Topology in Heterogeneous Quantum Networks

**Author:** David Vesterlund (Vesterlund Ventures, WestQuant Open)

---

Dear Editor,

I am submitting the manuscript "Memory Physics, Finite Capacity, and Topology in Heterogeneous Quantum Networks" for consideration as a research article in Quantum Reports.

The manuscript presents a systematic simulation study of how quantum-memory physics, finite memory capacity, and network topology jointly shape end-to-end entanglement delivery in heterogeneous quantum networks. The study addresses a gap in the existing literature: while analytical results characterize scaling and rate-memory trade-offs for specific homogeneous repeater architectures, and simulation frameworks enable detailed studies, no prior work systematically maps the interaction of representative memory profiles, finite capacity, topology classes, and heterogeneous resource placement within a single controlled framework.

The main contributions are:

1. A memory-topology atlas spanning six representative memory profiles, nine topology classes, and multiple traffic/noise/size conditions (36,450 runs), with a confirmatory subset of 8,730 runs.

2. Evidence that four candidate dimensionless scalar reductions ($\Pi$, $\Pi^*$, $\Pi_B$, $\Pi_{\rm phys}$) fail to provide a topology- and hardware-independent predictor of quality-qualified throughput (best pooled $R^2 = 0.177$).

3. A regime-dependent memory-budget threshold $m_c$ that varies from $m \leq 4$ to the grid boundary ($m = 64$) depending on memory profile, topology, and traffic.

4. A heterogeneous placement study showing that hub and bottleneck placement outperform peripheral placement in aggregate, with per-topology exceptions documented.

5. A harmonized cross-implementation robustness check using the SeQUeNCe and SimQN event frameworks, which reproduced the tested topology ordering and quality-qualified memory-profile ordering, while absolute performance levels and memory-capacity thresholds showed implementation sensitivity.

The study is positioned as a simulation study using representative memory models, not as an experimental comparison of complete hardware platforms or a validation against independent physics models. The cross-implementation check uses a common harmonized model on two event frameworks, not two independent physics replications. These limitations are stated explicitly in the manuscript.

The data and software are openly available at https://github.com/VesterlundCoder/quantum-network in the `paper_a_memory_physics_finite_capacity_topology` subdirectory, including simulator code, experiment configurations, random seeds, processed datasets, and reproduction scripts.

The author confirms that this manuscript is original, has not been published elsewhere, and is not under simultaneous consideration by another journal. The author has no external funding to declare and has disclosed the potential competing interest arising from the author's role as developer of the WestQuant simulator.

I look forward to your response.

Sincerely,

David Vesterlund
Vesterlund Ventures, WestQuant Open
david@vesterlundventures.se
