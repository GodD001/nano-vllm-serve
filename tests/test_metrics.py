"""observe_many must be indistinguishable from repeated Histogram.observe."""
import math
import random

from prometheus_client import CollectorRegistry, Histogram

from nanovllm.metrics import observe_many

BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)


def samples(registry):
    """name+labels -> value, without the creation timestamp (differs between instances)."""
    out = {}
    for metric in registry.collect():
        for s in metric.samples:
            if not s.name.endswith("_created"):
                out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def test_observe_many_matches_repeated_observe():
    rng = random.Random(0)
    values = [rng.choice([0.0, 0.001, 0.0049, 0.005, 0.02, 0.1, 0.7, 3.0]) for _ in range(200)]
    r1, r2 = CollectorRegistry(), CollectorRegistry()
    h1 = Histogram("h", "x", buckets=BUCKETS, registry=r1)
    h2 = Histogram("h", "x", buckets=BUCKETS, registry=r2)
    counts = {}
    for v in values:
        h1.observe(v)
        counts[v] = counts.get(v, 0) + 1
    for v, n in counts.items():
        observe_many(h2, v, n)
    observe_many(h2, 0.3, 0)   # no-op
    a, b = samples(r1), samples(r2)
    assert a.keys() == b.keys()
    for key in a:
        if key[0] == "h_sum":   # float addition order differs; equal up to rounding
            assert math.isclose(a[key], b[key], rel_tol=1e-12)
        else:                   # bucket counts and _count are exact
            assert a[key] == b[key], key
