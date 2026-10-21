# Sparse connected networks

The independent simulator, training dataset, checkpoints and institutional test
results remain the baseline. The optional connected mode is an exploratory
current-based spiking network. It does not implement a biological acquisition
system, gap junctions, plasticity or a BCI decoder.

## Model

The original Izhikevich Euler update and reset convention are retained:

```text
dv/dt = 0.04*v*v + 5*v + 140 - u + I_external + I_synaptic
du/dt = a*(b*v - u)
at v >= 30: the displayed sample is capped at 30; the next sample resets to c,
             and the next recovery value is u + d.
```

Each neuron has one synaptic role independent of its intrinsic RS/CH/FS
parameters. A seeded random subset of round(N * inhibitory_fraction) neurons is
inhibitory. All outgoing weights from one neuron have the same sign:

```text
excitatory source: w = strength
inhibitory source: w = -strength * inhibitory_gain
```

The network has fixed outgoing degree K = min(requested_degree, N - 1).
Each source samples K distinct, uniformly random destinations without itself.
In-degree varies. Connections are directed and may be reciprocal. A separate
seeded random stream preserves the baseline parameter and observation-noise
draws. Changing synaptic strength, gain, time constants or delay alone preserves
the wiring and roles. Changing the population, seed or inhibitory fraction can
change the wiring.

At each sampled time t:

1. Decay existing synaptic current by exp(-dt/tau).
2. Add every arriving presynaptic event to its target's current. Events sharing
   the same destination are summed, including excitation and inhibition.
3. Detect spikes from the current voltage and schedule their outgoing events.
4. Integrate with the current I_external + I_synaptic and apply the original
   reset rule. No next sample is integrated at the final stored time.

An event emitted at step s arrives at s + ceil(delay_ms/dt_ms), with at least
one step of delay. The UI reports the effective delay. Events arriving at or
beyond the end of the stored interval have no effect on its voltage traces.
Events arriving exactly at the final stored time update the current monitor,
but have no following voltage sample to affect. The last voltage sample is
kept as produced by the baseline solver and can exceed the spike threshold.

Spikes used for propagation, the connected raster, neuron spike counts and
population rates are detected before adding observation noise. Noise is applied
only to the stored voltage observations, after simulation. The existing
classifier retains its original preprocessing and eligibility rules.

With zero strength or zero outgoing degree, voltage traces and seeded noise are
bitwise equal to the independent baseline. Connected spike counts still use
actual threshold events rather than the baseline's observed >=29 mV samples.

## Inspection and interpretation

The Network tab includes:

- Total synapses, effective degree, E/I counts and delivered synaptic events.
- Full-population firing rate in bins of up to 5 ms, including the correct width
  of a shorter final bin. Hz means spikes per neuron per second within each bin.
- Mean signed synaptic current across all neurons, in model current units.
  Opposing excitatory and inhibitory inputs can cancel in this average.
- A 3-by-3 directed connection count between intrinsic parameter groups.
- Selected neuron's role, signed outgoing weight, in/out-degree and first 16
  destination IDs. All targets are retained, not just this preview.

The checkpoint sees voltage traces only. It is not retrained automatically and
has not been validated for synaptically driven signals. Coverage, confidence and
agreement remain observable, with explicit experimental labels. RS/CH/FS labels
identify the parameters used to generate a neuron, not a guarantee of a firing
pattern under every recurrent input or a biological cell identity.

The 10,000-neuron preset changes counts and synaptic settings. Current, seed and
variation controls remain as selected, so experiments can be compared. Use a
fixed seed when changing only strength, delay or inhibitory gain.

## Performance and cancellation

Destinations use int32 storage shaped (N, K). Each source has one float32 signed
weight and a boolean role; in-degrees use int32. Topology storage is approximately
4*N*K + 9*N + 72 bytes, excluding small Python/array headers. For 10,000 neurons
and 100 outgoing connections this is 4,090,072 bytes, plus 80 MB of voltage traces
at 200 ms and dt=0.1 ms. The application may retain old and new results together.
Spike event arrays, neuron parameters, predictions and model buffers add memory.

There is no N-by-N weight matrix or scan of every synapse on every timestep.
Only sources that fired send events. Construction and delivery use blocks of
256 sources; stepping checks cancellation every 32 timesteps. The existing
single worker discards stale results and retains only the latest pending job.

The budget is one billion synapses, one million neurons total and at most 1,000
outgoing connections per neuron. Defaults remain 30 neurons and 100 outgoing
connections (clamped to N - 1 for small populations); the preset remains 10,000
neurons. Above-budget requests fail before topology/traces allocation. This is
an application guard, not a measured hardware limit. High firing rates increase
event processing cost. Strong recurrent excitation or inhibition can produce
unusual dynamics; non-finite numerical results fail explicitly rather than being
silently clamped. There is no real-time performance guarantee.

Benchmark with the actual local checkpoint:

```powershell
python benchmark_population.py --connected --neurons 1000 10000 --out-degree 100 --threads 8 --output artifacts/network_benchmark.json
```

Use `--simulation-only` to omit the CNN. Parameters include `--weight`,
`--inhibitory-fraction`, `--inhibitory-gain`, `--delay-ms` and `--tau-ms`.
The JSON separates connected/independent configuration and records actual degree,
synapse count, graph bytes and delivered events. Timings include topology
construction, simulation, noise, inference when enabled, statistics and raster
preparation. They exclude rendering and retaining an old result.

## References and implementation choices

- E. M. Izhikevich, *Simple Model of Spiking Neurons* (2003),
  [author's publication page](https://www.izhikevich.org/publications/spikes.htm).
- Brian 2, [converting synaptic response functions to differential equations](https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html),
  including exponential decay and additive spike events.

The choice of fixed out-degree, shared outgoing magnitude, random role fraction,
uniform delay, limits and preset values is specific to this application. It is
not a claim to reproduce the original paper's complete network or a particular
biological circuit. Brian is a reference for the synaptic formulation, not a
runtime dependency.
