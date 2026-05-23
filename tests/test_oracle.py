"""Locks in oracle parsing semantics. The 'teach' patterns must strip the
article so 'this is a cup' yields class_name='cup', not 'a_cup'.
"""
from __future__ import annotations

import pytest

from cradle.oracle import RuleBasedOracle


@pytest.fixture(scope="module")
def oracle() -> RuleBasedOracle:
    return RuleBasedOracle()


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("this is a cup", "cup"),
        ("This is a Cup.", "cup"),
        ("that's a hand!", "hand"),
        ("it's an apple", "apple"),
        ("here's a wrench", "wrench"),
        ("this is the wrench", "wrench"),
        ("this is cup", "cup"),                 # no article
        ("this is a coffee mug", "coffee_mug"), # multi-word -> snake_case
        ("show me a phone", "phone"),
        ("show the keyboard", "keyboard"),
        ("look, a hat", "hat"),
    ],
)
def test_teach_patterns(oracle: RuleBasedOracle, utterance: str, expected: str) -> None:
    label = oracle.parse(utterance)
    assert label.action == "teach", f"{utterance!r} -> {label}"
    assert label.class_name == expected, f"{utterance!r} -> {label}"


@pytest.mark.parametrize("word", ["yes", "yeah", "right", "correct", "good"])
def test_positive_feedback(oracle: RuleBasedOracle, word: str) -> None:
    label = oracle.parse(word)
    assert label.action == "feedback"
    assert label.polarity == +1


@pytest.mark.parametrize("word", ["no", "nope", "wrong", "incorrect", "bad"])
def test_negative_feedback(oracle: RuleBasedOracle, word: str) -> None:
    label = oracle.parse(word)
    assert label.action == "feedback"
    assert label.polarity == -1


@pytest.mark.parametrize(
    "utterance,cmd",
    [("quit", "quit"), ("exit", "quit"), ("save", "save"), ("checkpoint", "save"),
     ("list", "list"), ("classes", "list")],
)
def test_commands(oracle: RuleBasedOracle, utterance: str, cmd: str) -> None:
    label = oracle.parse(utterance)
    assert label.action == "command"
    assert label.name == cmd


def test_forget_strips_article(oracle: RuleBasedOracle) -> None:
    label = oracle.parse("forget the cup")
    assert label.action == "command"
    assert label.name == "forget"
    assert label.arg == "cup"


@pytest.mark.parametrize("utterance", ["", "   ", "blah blah blah", "the cup is here"])
def test_noop(oracle: RuleBasedOracle, utterance: str) -> None:
    assert oracle.parse(utterance).action == "noop"
