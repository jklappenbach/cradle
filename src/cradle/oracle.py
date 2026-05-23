"""Oracle: utterance -> structured Label.

The oracle is the seam between human language and SPELA's integer class ids.
v1 is a small regex parser; v1.2 swaps in an LLM-backed parser with JSON-mode
output. Same Label schema either way, so main.py never changes.

Label shape:
  {"action": "teach",    "class_name": str}                 - label recent frames
  {"action": "feedback", "polarity": +1 | -1}               - reinforce / penalize
  {"action": "command",  "name": "save"|"forget"|"quit"|"list", "arg": str|None}
  {"action": "noop"}                                          - unparsed
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Label:
    action: str
    class_name: Optional[str] = None
    polarity: Optional[int] = None
    name: Optional[str] = None
    arg: Optional[str] = None


class Oracle:
    def parse(self, utterance: str) -> Label:
        raise NotImplementedError


class RuleBasedOracle(Oracle):
    """Regex-based parser. Patterns covered:

      "this is a cup" / "that's a cup" / "it's a cup"  -> teach(cup)
      "this is coffee_mug"                              -> teach(coffee_mug)
      "show me a cup" / "look, a cup"                   -> teach(cup)
      "yes" / "right" / "good" / "correct"              -> feedback(+1)
      "no" / "wrong" / "bad" / "incorrect"              -> feedback(-1)
      "save" / "save model" / "checkpoint"              -> command(save)
      "list" / "classes" / "what do you know"           -> command(list)
      "forget cup"                                      -> command(forget, "cup")
      "quit" / "exit" / "stop" / "bye"                  -> command(quit)
    """

    # The required parts (greeting + verb) match first; the article + name
    # tail is shared. Keep the article group required-and-consume-its-space
    # so "this is a cup" yields name="cup" not "a cup".
    _TEACH = re.compile(
        # "this is", "that's", "it's", "here is", "here's"
        r"^(?:(?:this|that|it)(?:\s+is|'?s)|here(?:'?s|\s+is))"
        r"\s+(?:(?:a|an|the)\s+)?(?P<name>[a-z][a-z0-9 _-]{0,40}?)\s*[.!?]*$",
        re.IGNORECASE,
    )
    _SHOW = re.compile(
        r"^(?:show(?:\s+me)?|see|look(?:,)?)"
        r"\s+(?:(?:a|an|the)\s+)?(?P<name>[a-z][a-z0-9 _-]{0,40}?)\s*[.!?]*$",
        re.IGNORECASE,
    )
    _FORGET = re.compile(
        r"^forget\s+(?:(?:a|an|the)\s+)?(?P<name>[a-z][a-z0-9 _-]{0,40}?)\s*[.!?]*$",
        re.IGNORECASE,
    )

    _POS_WORDS = {"yes", "yeah", "yep", "correct", "right", "good", "true"}
    _NEG_WORDS = {"no", "nope", "wrong", "incorrect", "bad", "false"}
    _SAVE_WORDS = {"save", "save model", "checkpoint"}
    _LIST_WORDS = {"list", "classes", "what do you know", "what have you learned"}
    _QUIT_WORDS = {"quit", "exit", "stop", "bye"}

    def parse(self, utterance: str) -> Label:
        s = utterance.strip().lower().rstrip(".!?")
        if not s:
            return Label(action="noop")

        if s in self._QUIT_WORDS:
            return Label(action="command", name="quit")
        if s in self._SAVE_WORDS:
            return Label(action="command", name="save")
        if s in self._LIST_WORDS:
            return Label(action="command", name="list")
        if s in self._POS_WORDS:
            return Label(action="feedback", polarity=+1)
        if s in self._NEG_WORDS:
            return Label(action="feedback", polarity=-1)

        m = self._FORGET.match(s)
        if m:
            return Label(action="command", name="forget",
                         arg=re.sub(r"\s+", "_", m.group("name").strip()))

        for pat in (self._TEACH, self._SHOW):
            m = pat.match(s)
            if m:
                name = re.sub(r"\s+", "_", m.group("name").strip())
                return Label(action="teach", class_name=name)

        return Label(action="noop")


class LLMOracle(Oracle):
    """Placeholder. v1.2 will plug in a local LLM (llama.cpp / Ollama) using
    JSON-mode output to parse arbitrary phrasings into Label. The interface
    above is intentionally the same as RuleBasedOracle so main.py doesn't
    change when we swap.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LLMOracle is a v1.2 seam; use RuleBasedOracle for v1.")
