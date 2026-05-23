from __future__ import annotations

import pytest

from cradle.registry import LabelRegistry, RegistryFull


def test_get_or_create_assigns_sequential_ids():
    r = LabelRegistry(capacity=8)
    assert r.get_or_create("cup") == 0
    assert r.get_or_create("hand") == 1
    assert r.get_or_create("cup") == 0  # idempotent
    assert len(r) == 2


def test_name_lookup_round_trips():
    r = LabelRegistry(capacity=8)
    cid = r.get_or_create("Coffee Mug")
    assert r.name(cid) == "coffee mug"  # normalized to lowercase
    assert r.name(99) is None


def test_capacity_exhaustion_raises():
    r = LabelRegistry(capacity=2)
    r.get_or_create("a")
    r.get_or_create("b")
    with pytest.raises(RegistryFull):
        r.get_or_create("c")


def test_state_dict_round_trip():
    r = LabelRegistry(capacity=8)
    r.get_or_create("cup")
    r.get_or_create("hand")
    state = r.state_dict()

    r2 = LabelRegistry(capacity=2)  # different capacity; load should overwrite
    r2.load_state_dict(state)
    assert r2.capacity == 8
    assert r2.known_names() == ["cup", "hand"]
    assert r2.get_or_create("cup") == 0  # mapping preserved


def test_empty_name_rejected():
    r = LabelRegistry(capacity=4)
    with pytest.raises(ValueError):
        r.get_or_create("   ")
