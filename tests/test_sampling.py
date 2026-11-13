"""SamplingParams validation and the batched sampler (CPU torch)."""
import pytest

from nanovllm.sampling_params import SamplingParams

torch = pytest.importorskip("torch")
from nanovllm.layers import sampler as sampler_mod  # noqa: E402
from nanovllm.layers.sampler import Sampler, SamplingTensors, apply_top_k_top_p  # noqa: E402


def test_defaults_match_upstream():
    sp = SamplingParams()
    assert (sp.temperature, sp.max_tokens, sp.ignore_eos) == (1.0, 64, False)
    assert (sp.top_k, sp.top_p, sp.stop_token_ids) == (-1, 1.0, [])
    assert not sp.greedy


@pytest.mark.parametrize("kwargs", [dict(temperature=-0.1), dict(max_tokens=0), dict(top_k=0),
                                    dict(top_p=0.0), dict(top_p=1.5)])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_zero_temperature_is_greedy():
    assert SamplingParams(temperature=0).greedy
    assert SamplingParams(temperature=1e-7).greedy
    assert not SamplingParams(temperature=0.01).greedy


def _logits(batch=4, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, vocab, generator=g)


def test_default_batch_uses_upstream_kernel(monkeypatch):
    calls = []

    def spy(logits, temperatures):
        calls.append(temperatures)
        return logits.argmax(-1)

    monkeypatch.setattr(sampler_mod, "_upstream_sample", spy)
    s = Sampler()
    logits = _logits()
    s(logits, SamplingTensors(temperatures=torch.ones(4)))
    s(logits, torch.ones(4))  # upstream call signature still works
    assert len(calls) == 2


def test_all_greedy_is_argmax(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_upstream_sample", lambda *a: pytest.fail("should not sample"))
    logits = _logits()
    out = Sampler()(logits, SamplingTensors(temperatures=None, all_greedy=True))
    assert torch.equal(out, logits.argmax(-1))


def test_mixed_greedy_rows_are_argmax():
    torch.manual_seed(0)
    logits = _logits(batch=6, vocab=1000)
    greedy = torch.tensor([True, False, True, False, True, False])
    for _ in range(20):
        out = Sampler()(logits, SamplingTensors(temperatures=torch.full((6,), 5.0), greedy=greedy))
        assert torch.equal(out[greedy], logits.argmax(-1)[greedy])


def test_top_k_1_and_tiny_top_p_are_argmax():
    torch.manual_seed(0)
    logits = _logits(batch=8, vocab=500)
    t = torch.full((8,), 3.0)
    for _ in range(20):
        a = Sampler()(logits, SamplingTensors(temperatures=t, top_ks=torch.ones(8, dtype=torch.int64)))
        b = Sampler()(logits, SamplingTensors(temperatures=t, top_ps=torch.full((8,), 1e-6)))
        assert torch.equal(a, logits.argmax(-1))
        assert torch.equal(b, logits.argmax(-1))


def test_top_k_samples_stay_in_top_k():
    torch.manual_seed(0)
    logits = _logits(batch=3, vocab=200)
    ks = torch.tensor([1, 5, 200])
    allowed = [set(logits[i].topk(int(k)).indices.tolist()) for i, k in enumerate(ks)]
    seen = [set() for _ in range(3)]
    for _ in range(300):
        out = Sampler()(logits, SamplingTensors(temperatures=torch.full((3,), 10.0), top_ks=ks))
        for i in range(3):
            assert out[i].item() in allowed[i]
            seen[i].add(out[i].item())
    assert len(seen[1]) == 5  # high temperature: all 5 allowed tokens appear


def test_top_p_nucleus_mask():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    masked = apply_top_k_top_p(logits.clone(), None, torch.tensor([0.75]))
    kept = torch.isfinite(masked[0]).nonzero().flatten().tolist()
    assert kept == [0, 1]          # 0.5 + 0.3 >= 0.75
    masked = apply_top_k_top_p(logits.clone(), None, torch.tensor([0.81]))
    assert torch.isfinite(masked[0]).nonzero().flatten().tolist() == [0, 1, 2]
    masked = apply_top_k_top_p(logits.clone(), torch.tensor([3]), torch.tensor([1.0]))
    assert torch.isfinite(masked[0]).nonzero().flatten().tolist() == [0, 1, 2]


def test_top_k_then_top_p_per_row():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0],
                           [0.0, 1.0, 2.0, 3.0, 4.0]])
    masked = apply_top_k_top_p(logits.clone(), torch.tensor([2, 5]), torch.tensor([1.0, 0.5]))
    assert torch.isfinite(masked[0]).tolist() == [True, True, False, False, False]
    assert torch.isfinite(masked[1]).tolist()[4] and not torch.isfinite(masked[1]).tolist()[0]
