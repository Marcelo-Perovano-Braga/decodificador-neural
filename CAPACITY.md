# Capacity estimates for a Ryzen 7 5700X3D and 32 GiB RAM

These are planning estimates, not measured limits of that computer. The application
supports independent neurons and an optional sparse connected mode. The measured
optimization results below concern the independent baseline. Connected network
equations, current storage and measurement instructions are in [NETWORKS.md](NETWORKS.md).

The 5700X3D has eight cores and sixteen threads
([AMD specifications](https://ir.amd.com/news-events/press-releases/detail/1177/amd-reveals-next-gen-desktop-processors-for-extreme-pc-gaming-and-creator-performance)).
The application now uses up to eight CPU inference threads. Its configurable
population cap is 1,000,000 total, distributed freely among RS, CH and FS.
This expanded configuration allowance is not a measured hardware limit or a
full-scale benchmark result; the older measurements below retain their original scope.

## Independent neurons

At the default 200 ms duration and 0.1 ms timestep, every neuron retains 2,000
float32 voltage samples:

```text
one population:            2,000 × 4 × N = 8,000 × N bytes
old + replacement result:                16,000 × N bytes
```

The interface retains its previous result while calculating the next. Model
buffers, Python prediction objects, parameters, labels, and Qt add memory beyond
these trace arrays. Noise and validation temporaries are now bounded rather than
proportional to the entire population.

| Neurons | One full trace array | Two populations' trace arrays |
|---:|---:|---:|
| 100,000 | 0.75 GiB | 1.49 GiB |
| 300,000 | 2.24 GiB | 4.47 GiB |
| 1,000,000 | 7.45 GiB | 14.90 GiB |
| 1,500,000 | 11.18 GiB | 22.35 GiB |

Reserving roughly 8 GiB of a 32 GiB system for Windows, other applications,
runtime/model buffers, and headroom leaves an illustrative 24 GiB data budget.
Allowing another approximately 512 bytes per neuron for result objects and
metadata gives a rough ceiling of 1.56 million independent neurons. A conservative
planning range is **1–1.5 million, limited by memory**. The application permits
up to one million; its runtime and peak memory at that scale require local
measurement. This range does not promise interactive update speed.

Simulation duration matters directly: increasing the window from 0.2 to 1 second
multiplies trace storage and stepping work by five. Reducing the timestep has a
similar effect. The current update preserves both settings and every trace.

## Connected neurons: topology determines capacity

The connected implementation uses int32 targets and one shared outgoing weight
per source, approximately `4*N*k + 9*N + 72` bytes for topology. Old and new
populations can retain separate graphs. It has a one-billion-synapse application
budget; the 10,000-neuron / 100-outgoing preset has one million synapses. Actual
spike event arrays add storage proportional to activity. These configured limits
are below the hypothetical memory-only ceilings in the comparison below.

The following calculations retain the same full voltage traces and the previous
result. They compare hypothetical representations with an individual weight per
edge and one shared connectivity graph. They are not the memory accounting of
the connected implementation above. Retaining two such graphs would increase
their requirement further.

For a sparse network with average outgoing degree `k`, one simple representation
stores a float32 weight and a 32-bit target index per edge:

```text
edges E ≈ N × k
minimal edge storage ≈ 8 × N × k bytes
data budget ≈ (16,000 + 512 + 8 × k) × N bytes
```

| Representation | Approximate 24 GiB memory ceiling | Main limitation |
|---|---:|---|
| Independent, full traces | 1.56 million | Trace retention and CNN throughput |
| Sparse, 100 outgoing synapses/neuron | 1.49 million | Delivery of spike events and graph storage |
| Sparse, 1,000 outgoing synapses/neuron | 1.05 million | Many more events and connections |
| Dense, one float32 weight per possible directed pair | About 78,000 | Quadratic storage and computation |

These are **memory-only ceilings under simplified assumptions**, not usable
real-time population counts. A first connected-network target should be much
smaller: for example, 10,000 neurons with 100 outgoing connections each (about
one million synapses), followed by measurement and gradual scaling.

For dense weights, storage alone is approximately `4 × N²` bytes: 50,000 neurons
require 10 GB in weights, and 100,000 require 40 GB before voltage traces and other
state. The dense ceiling above solves `4 × N² + 16,000 × N ≈ 24 GiB`; reserving
more working space puts a planning bound nearer 70,000–75,000. Full dense updates
at every 0.1 ms step would also impose substantial computation.

Synaptic delays, conductance states, event queues, plasticity traces, and recording
synaptic variables increase memory use. More importantly, spike-delivery work
depends on firing rates and connectivity: an event-driven implementation delivers
approximately `N × k × average_firing_rate` synaptic events per simulated second.
Sparse event-driven methods and dense timestep-based multiplication have very
different performance, so a single count of "connected neurons" is insufficient.

The existing classifier was trained on independent-neuron signals. Its accuracy
on synaptically driven signals would need a separate evaluation.

## RX 6600

The RX 6600 has a separate 8 GB of VRAM
([AMD specifications](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6600.html)).
This is not an extra 8 GB in the CPU's ordinary RAM pool. The current application
and population benchmark execute on the CPU; this update does not assume or
enable an untested AMD GPU backend. A GPU implementation would need its own
compatibility and performance validation on the Windows installation. Peak GPU
TFLOPS do not directly predict neuron throughput.

## Measure this computer

Using the environment and trained checkpoint already used by the application:

```powershell
python benchmark_population.py --neurons 3000 10000 100000 --threads 8 --output artifacts/population_benchmark_8threads.json
python benchmark_population.py --neurons 3000 10000 100000 --threads 4 --output artifacts/population_benchmark_4threads.json
```

Each size uses three measured runs after warm-up. Times include simulation,
observation noise, inference, statistics, and raster preparation, with no Qt
rendering or retained prior population. Compare with Task Manager's process
memory during a second large update in the GUI to measure actual interactive
working memory.

The same override is available in the application: `python main_app.py --threads 4`
or `ClassificadorIzhikevich.exe --threads 4`. With no override it uses up to eight
threads, leaving at least one logical CPU unused by its inference pool when possible.

For a completed 200 ms window, processing must finish within 0.2 seconds to match
the simulated duration. A responsive window while work runs in the background
is a separate property. Extending this application into continuous streaming
would also require persistent neuron state and incremental processing.

## Validation of this optimization

A before/after comparison used fresh Linux processes on an AMD EPYC 9V74 cloud
environment with nine visible logical CPUs and an eight-CPU quota. Each run
simulated and classified 100,000 neurons for 200 ms, with a 0.1 ms timestep,
stimulus 30, seed 42, jitter/noise enabled, and inference batches of 64. The old
application's one inference thread was compared with the optimized eight-thread
runtime, including convolution/BatchNorm fusion. Each version had three measured
runs after model warm-up; timing excluded startup and GUI rendering.

| Measurement | Before | Optimized |
|---|---:|---:|
| Median total processing time | 31.305 s | 14.147 s |
| Maximum peak process RSS across those runs | 3.27 GiB | 1.07 GiB |

This is approximately 2.2 times the processing throughput and 67% less peak
process memory in this environment. It is not a prediction of exact Windows
timings, nor does process RSS include keeping an earlier GUI result in memory.

The offscreen Qt GUI also processed the configured maximum of 300,000 neurons
with full inference, all table rows, a raster sample of 200, and inspection of
the last neuron's complete 2,000-sample trace. That run took 47.3 seconds on the
cloud CPU. A 10 ms event-loop timer continued firing, with a maximum observed
gap of 135 ms. Cancelling another large update and replacing it with a
1,000-neuron request completed in about 0.2 seconds. Dark and light layouts were
checked at the supported minimum window size, and the worker shut down cleanly.

These timings used an explicitly **untrained, known-output CNN fixture** to
exercise the actual architecture and application. They do not measure the
scientific accuracy of the private trained checkpoint. Regression tests compare
the optimized Izhikevich and LIF solvers bit-for-bit with the previous stepping
algorithm, verify the original seeded noise and subsequent RNG state, and check
model outputs after fusion/parallelism within float32 tolerance. Native Windows
packaging and behavior with the actual trained checkpoint must be checked on
the Windows computer.
