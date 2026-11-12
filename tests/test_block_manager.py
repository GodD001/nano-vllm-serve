"""Block manager invariants, driven the same way the upstream scheduler drives it.

Protocol (from scheduler.py / llm_engine.py):
  admit:    can_allocate(seq) -> allocate(seq, n_cached)
  prefill:  num_scheduled_tokens = k ; hash_blocks ; num_cached_tokens += k   (possibly in chunks)
  decode:   can_append / may_append ; num_scheduled_tokens = 1 ; hash_blocks ; num_cached += 1 ; append_token
  preempt / finish: deallocate(seq)
"""
import random

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from tests.fake_engine import check_block_manager


def _prefill_chunk(bm, seq, k):
    seq.num_scheduled_tokens = k
    bm.hash_blocks(seq)
    seq.num_cached_tokens += k
    seq.num_scheduled_tokens = 0


def _admit(bm, seq):
    n_cached = bm.can_allocate(seq)
    if n_cached == -1:
        return False
    bm.allocate(seq, n_cached)
    return True


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("block_size", [4, 16])
def test_randomized_ops_keep_invariants(seed, block_size):
    rng = random.Random(seed)
    Sequence.block_size = block_size
    num_blocks = rng.choice([8, 16, 32])
    bm = BlockManager(num_blocks, block_size)

    # A few shared prefixes so the prefix cache actually gets hit.
    prefixes = [[rng.randrange(20) for _ in range(rng.randrange(1, 4 * block_size))] for _ in range(3)]

    def new_tokens():
        p = rng.choice(prefixes) if rng.random() < 0.7 else []
        return p + [rng.randrange(20) for _ in range(rng.randrange(1, 2 * block_size))]

    prefilling: list[Sequence] = []
    decoding: list[Sequence] = []
    evicted: list[Sequence] = []  # preempted, waiting to be re-admitted (recompute)

    def live():
        return prefilling + decoding

    for _ in range(400):
        op = rng.random()
        if op < 0.2:
            seq = evicted.pop(0) if evicted and rng.random() < 0.5 else Sequence(new_tokens())
            if _admit(bm, seq):
                prefilling.append(seq)
            else:
                evicted.append(seq)
        elif op < 0.45 and prefilling:
            seq = rng.choice(prefilling)
            remaining = seq.num_tokens - seq.num_cached_tokens
            _prefill_chunk(bm, seq, rng.randint(1, remaining))
            if seq.num_cached_tokens == seq.num_tokens:
                seq.append_token(rng.randrange(20))
                prefilling.remove(seq)
                decoding.append(seq)
        elif op < 0.8 and decoding:
            seq = rng.choice(decoding)
            if bm.can_append(seq):
                bm.may_append(seq)
                _prefill_chunk(bm, seq, 1)
                seq.append_token(rng.randrange(20))
            else:  # out of blocks: preempt, like Scheduler.preempt
                bm.deallocate(seq)
                decoding.remove(seq)
                evicted.append(seq)
        elif op < 0.9 and live():
            seq = rng.choice(live())  # preempt from any state, incl. mid-chunked-prefill
            bm.deallocate(seq)
            (prefilling if seq in prefilling else decoding).remove(seq)
            evicted.append(seq)
        elif decoding:
            seq = rng.choice(decoding)  # finish
            bm.deallocate(seq)
            decoding.remove(seq)
        check_block_manager(bm, live())
        for seq in prefilling:
            assert len(seq.block_table) == seq.num_blocks
        for seq in decoding:
            # The newest token's KV is not written yet; its block is allocated
            # lazily by may_append at the next decode schedule.
            assert seq.num_cached_tokens == seq.num_tokens - 1
            assert len(seq.block_table) == -(-seq.num_cached_tokens // block_size)

    for seq in live():
        bm.deallocate(seq)
    check_block_manager(bm, [])
    assert len(bm.free_block_ids) == num_blocks
    assert not bm.used_block_ids


def test_shared_prefix_reuses_blocks_while_live():
    bs = 4
    Sequence.block_size = bs
    bm = BlockManager(16, bs)
    prefix = list(range(1, 3 * bs + 1))  # 3 full blocks
    a = Sequence(prefix + [100, 101])
    assert _admit(bm, a)
    _prefill_chunk(bm, a, a.num_tokens)

    b = Sequence(prefix + [200])
    n_cached = bm.can_allocate(b)
    assert n_cached == 3
    bm.allocate(b, n_cached)
    assert b.block_table[:3] == a.block_table[:3]
    assert b.num_cached_tokens == 3 * bs
    assert b.block_table[3] != a.block_table[3]
    for bid in a.block_table[:3]:
        assert bm.blocks[bid].ref_count == 2
    check_block_manager(bm, [a, b])

    bm.deallocate(a)
    for bid in b.block_table[:3]:
        assert bm.blocks[bid].ref_count == 1
    check_block_manager(bm, [b])
    bm.deallocate(b)
    check_block_manager(bm, [])
    assert len(bm.free_block_ids) == 16


def test_prefix_cache_revives_freed_blocks():
    bs = 4
    Sequence.block_size = bs
    bm = BlockManager(16, bs)
    prefix = list(range(1, 2 * bs + 1))
    a = Sequence(prefix + [7])
    assert _admit(bm, a)
    _prefill_chunk(bm, a, a.num_tokens)
    a_blocks = list(a.block_table[:2])
    bm.deallocate(a)  # blocks go back to the free list but keep their hashes

    b = Sequence(prefix + [8, 9])
    n_cached = bm.can_allocate(b)
    assert n_cached == 2
    bm.allocate(b, n_cached)
    assert b.block_table[:2] == a_blocks
    check_block_manager(bm, [b])


def test_only_full_blocks_are_cached():
    bs = 4
    Sequence.block_size = bs
    bm = BlockManager(16, bs)
    a = Sequence(list(range(1, bs + 3)))  # 1 full block + 2 tokens
    assert _admit(bm, a)
    _prefill_chunk(bm, a, a.num_tokens)
    assert bm.blocks[a.block_table[0]].hash != -1
    assert bm.blocks[a.block_table[1]].hash == -1  # partial block never hashed
    assert len(bm.hash_to_block_id) == 1


def test_divergent_suffix_does_not_hit_cache():
    bs = 4
    Sequence.block_size = bs
    bm = BlockManager(16, bs)
    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert _admit(bm, a)
    _prefill_chunk(bm, a, a.num_tokens)
    b = Sequence([1, 2, 3, 4, 5, 6, 7, 0, 9])  # differs in block 1
    assert bm.can_allocate(b) == 1
