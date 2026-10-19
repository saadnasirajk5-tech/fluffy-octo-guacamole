"""
text_splitters
==============

A small, production-shaped text-splitting library — the engine behind
what LangChain calls CharacterTextSplitter / RecursiveCharacterTextSplitter,
rebuilt the way a library author would structure it.

ARCHITECTURE (three layers, top depends on bottom, never the reverse)
=====================================================================

    ┌─────────────────────────────────────────────────────────────┐
    │  LAYER 3 — PUBLIC API (what a user imports)                  │
    │                                                              │
    │    CharacterTextSplitter        → one separator, one level   │
    │    RecursiveCharacterTextSplitter → separator ladder +       │
    │                                    recursion on big pieces   │
    └───────────────────────────┬─────────────────────────────────┘
                                │ inherits from
    ┌───────────────────────────▼─────────────────────────────────┐
    │  LAYER 2 — CONTRACT (BaseTextSplitter)                       │
    │                                                              │
    │    • owns + validates all configuration ONCE                 │
    │    • defines the interface every splitter must implement     │
    │    • offers a convenience adapter for the RAG notebook      │
    └───────────────────────────┬─────────────────────────────────┘
                                │ is built from
    ┌───────────────────────────▼─────────────────────────────────┐
    │  LAYER 1 — ENGINE (private, stateless functions)             │
    │                                                              │
    │    _split_with_offsets()  → CUT text on a separator while    │
    │                              remembering where each piece   │
    │                              came from                      │
    │    _merge_with_overlap() → PACK pieces greedily into        │
    │                              chunks + seed overlap tails     │
    └───────────────────────────┬─────────────────────────────────┘
                                │ produces / consumes
    ┌───────────────────────────▼─────────────────────────────────┐
    │  LAYER 0 — DATA MODEL                                        │
    │                                                              │
    │    Split — an immutable value object: a piece of text PLUS  │
    │    its exact position in the original document.              │
    │    One invariant is enforced at construction time:           │
    │        original[char_start:char_end] == text                 │
    └─────────────────────────────────────────────────────────────┘

DESIGN DECISIONS (the "why")
============================

1. WHY OFFSETS?  A chunk that knows where it came from can be cited
   verbatim, merged with its neighbours, or highlighted in a UI.
   Most chunkers throw this information away; we keep it end-to-end.

2. WHY AN ABC?  Every splitter (character, recursive, future token /
   markdown / code splitters) shares configuration and validation.
   Putting that in BaseTextSplitter means "add a new splitter" costs
   one class, not a copy-paste of the validation logic.

3. WHY STATELESS ENGINE FUNCTIONS?  They take inputs and return
   outputs with zero side effects → trivially unit-testable, reusable
   by both public classes, and safe to reason about.

4. WHY injectable length_function?  Size decisions (does it fit in
   chunk_size?) are decoupled from character counting. Pass `len` for
   characters, or a tiktoken-based counter for tokens — the engine
   never changes, only the measuring stick does. Offsets ALWAYS stay
   in characters, because they are positions, not sizes.

5. WHY greedy merging?  Splitting produces many small pieces; packing
   them back-to-back until full gives, in one linear pass, chunks that
   are both ~chunk_size and aligned to natural boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["Split", "BaseTextSplitter", "CharacterTextSplitter", "RecursiveCharacterTextSplitter"]


# ========================================================================== #
# LAYER 0 — DATA MODEL                                                       #
# ========================================================================== #


@dataclass(frozen=True)
class Split:
    """A piece of text that remembers exactly where it came from.

    This is the only thing that travels between the layers. Keeping it
    immutable means functions can freely share/hold references without
    any layer accidentally mutating another layer's data.

    Attributes:
        text:       The piece itself.
        char_start: Offset of the first character in the ORIGINAL document.
        char_end:   Offset one past the last character (exclusive — Python
                    slice convention, so source[char_start:char_end] works).
    """

    text: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        """Enforce the invariant at construction time.

        If every Split in the system is a verbatim slice of the source,
        then citation integrity is guaranteed by construction and we
        never need to re-check it later — except as an assertion in tests.
        """
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError(
                f"Invalid offsets: 0 <= char_start <= char_end violated "
                f"(got start={self.char_start}, end={self.char_end})"
            )
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError(
                "Split invariant violated: text must equal "
                "source[char_start:char_end] — a Split may only represent "
                "a contiguous, unmodified span of the original document."
            )

    # -- convenience ------------------------------------------------------- #

    def is_verbatim_slice_of(self, source: str) -> bool:
        """True if `source` really does contain this exact span."""
        return source[self.char_start : self.char_end] == self.text

    def to_dict(self) -> dict:
        """Export to the chunk schema used by our RAG notebook."""
        return {"text": self.text, "char_start": self.char_start, "char_end": self.char_end}


# ========================================================================== #
# LAYER 1 — ENGINE (private, stateless)                                     #
# ========================================================================== #


def _split_with_offsets(text: str, sep: str, base: int = 0) -> list[Split]:
    """Cut `text` wherever `sep` occurs, remembering each piece's position.

    Args:
        text: The text to cut (may be a sub-piece of a bigger document).
        sep:  The separator to cut on. The special value "" means "cut
              between every character" — the last-resort strategy used
              by the recursive splitter when nothing else fits.
        base: Offset of `text` inside the ORIGINAL document, so offsets
              come out absolute even when we recurse into sub-pieces.
              This is how the recursive splitter keeps positions valid.

    Returns:
        Splits for the non-separator content. The separators themselves
        belong to no piece — but because every piece knows its absolute
        start, re-joining consecutive pieces with `sep` reproduces the
        original span exactly. Offsets therefore survive round-trips.
    """
    if text == "":
        return []
    if sep == "":  # last resort: split into single characters
        return [Split(text[i], base + i, base + i + 1) for i in range(len(text))]

    pieces: list[Split] = []
    start = 0
    while True:
        idx = text.find(sep, start)
        if idx == -1:  # no more separators — the remainder is the last piece
            pieces.append(Split(text[start:], base + start, base + len(text)))
            return pieces
        pieces.append(Split(text[start:idx], base + start, base + idx))
        start = idx + len(sep)  # jump PAST the separator


def _assemble(current: Sequence[Split], sep: str) -> Split:
    """Join a run of consecutive pieces back into one chunk Split.

    Consecutive pieces in `current` are always adjacent in the original
    (the engine only ever groups neighbours), so re-inserting `sep`
    between them reconstructs the original bytes byte-for-byte. That is
    why the new Split still satisfies the verbatim invariant: its span
    [first.char_start, last.char_end] contains exactly those pieces and
    the separators between them.
    """
    text = sep.join(s.text for s in current)
    return Split(text, current[0].char_start, current[-1].char_end)


def _merge_with_overlap(
    splits: Sequence[Split],
    sep: str,
    chunk_size: int,
    overlap: int,
    length: Callable[[str], int],
) -> list[Split]:
    """Greedily pack pieces into chunks no larger than chunk_size.

    This function is the heart of every chunker in this module.

    Greedy algorithm, one linear pass:
      * Keep adding pieces to the open chunk while they fit.
      * The moment a piece WOULD overflow the chunk, close the chunk
        and start a new one.
      * Before starting the new chunk, seed it with the TAIL of the
        chunk we just closed (~`overlap` worth of content, measured
        backwards piece-by-piece, never cutting a piece in half).

    Why seed the tail?  If a sentence straddles a boundary, the overlap
    guarantees it appears intact (with its context) in at least one
    chunk, so retrieval never returns a context-free fragment.

    Args:
        splits:     Pieces to pack. Must be consecutive in the source.
        sep:        Separator to re-insert between pieces when joining.
        chunk_size: Maximum size of a chunk, as measured by `length`.
        overlap:    Target size of the shared tail between chunks.
        length:     The measuring stick — `len` for characters, or any
                    callable (e.g. a token counter). NOTE: `length` only
                    drives DECISIONS; the text produced is unaffected,
                    so char offsets always stay correct.

    Returns:
        Chunks (as Splits) with real content — whitespace-only results
        are filtered out, so callers never see empty chunks.
    """
    sep_len = length(sep)
    chunks: list[Split] = []
    current: list[Split] = []  # pieces of the chunk being built
    current_len = 0            # size of `current` under `length`

    for piece in splits:
        # Cost of admitting this piece: its own size, plus one separator
        # if it won't be the first piece (the join re-inserts `sep`).
        add_len = length(piece.text) + (sep_len if current else 0)

        # Overflow check: would this piece push the open chunk past the
        # limit? If so, CLOSE the chunk first, THEN prepare a new one.
        if current and current_len + add_len > chunk_size:
            chunks.append(_assemble(current, sep))

            # Build the overlap tail: walk BACKWARDS over the pieces we
            # just emitted and keep as many as fit within `overlap`.
            # Pieces are never cut — overlap is approximate by design,
            # which keeps every boundary on a natural separator.
            tail: list[Split] = []
            tail_len = 0
            for prev in reversed(current):
                cost = length(prev.text) + (sep_len if tail else 0)
                if tail_len + cost > overlap:
                    break
                tail.insert(0, prev)  # prepend: preserve reading order
                tail_len += cost
            current, current_len = tail, tail_len

            # Edge case: one huge piece (e.g. a giant sentence) that not
            # even tail + piece can hold. Drop the tail so this piece
            # becomes a clean chunk of its own. Without this guard we
            # would emit a chunk that is pure overlap-duplicate.
            if current_len + add_len > chunk_size:
                current, current_len = [], 0

        current.append(piece)
        current_len += add_len

    if current:  # don't forget the final, never-overflowed chunk
        chunks.append(_assemble(current, sep))

    return [c for c in chunks if c.text.strip()]


# ========================================================================== #
# LAYER 2 — CONTRACT                                                         #
# ========================================================================== #


class BaseTextSplitter(ABC):
    """Configuration, validation, and the interface for all splitters.

    Every concrete splitter MUST implement `split_text`. Configuration
    lives here so validation happens exactly ONCE, in ONE place — a new
    splitter subclass can never forget it.

    Args:
        chunk_size:       Maximum chunk size, measured by `length_function`.
        overlap:          Target shared-tail size between chunks.
        length_function:  How "size" is measured. `len` → characters.
                          Swap in a token counter (e.g. tiktoken) and the
                          whole engine silently switches to token budgets.

    Raises:
        ValueError: on impossible configuration, at construction time —
                    a professional library fails FAST, at the boundary,
                    not silently mid-pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 80,
        length_function: Callable[[str], int] = len,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            # overlap >= chunk_size would mean the window never advances
            # -> infinite loop. Reject it up front, with a clear message.
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < chunk_size "
                f"(got overlap={overlap}, chunk_size={chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.length_function = length_function

    # -- the one method every splitter must provide ------------------------ #

    @abstractmethod
    def split_text(self, text: str) -> list[Split]:
        """Cut `text` into chunks. Returns Splits with absolute offsets."""

    # -- convenience adapters ---------------------------------------------- #

    def split_chunks(self, text: str) -> list[dict]:
        """Notebook-friendly output: the same chunk dicts we used in RAG.

        Keeping the conversion HERE (instead of in every splitter) is the
        Adapter pattern: the engine speaks `Split`, downstream code speaks
        plain dicts, and this one method is the translator.
        """
        return [s.to_dict() for s in self.split_text(text)]


# ========================================================================== #
# LAYER 3 — PUBLIC API                                                       #
# ========================================================================== #


class CharacterTextSplitter(BaseTextSplitter):
    """Split on ONE separator, then greedily merge. No fallback strategy.

    Strategy (flat, one level):
        split on `separator` → merge pieces until chunk_size → overlap.

    Use when the input's structure is dominated by a single delimiter
    (paragraphs in prose, records in a log, rows in a CSV).

    Known limitation (by design): a piece LARGER than chunk_size is
    emitted as an oversized chunk, because there is no plan B. If you
    need a fallback, use RecursiveCharacterTextSplitter.

    Args:
        separator: The single delimiter to cut on (default: paragraph
                   breaks, "\\n\\n").

    Example:
        >>> splitter = CharacterTextSplitter(separator="\\n\\n", chunk_size=500, overlap=80)
        >>> chunks = splitter.split_chunks(doc)   # list of dicts, like before
    """

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 500,
        overlap: int = 80,
        length_function: Callable[[str], int] = len,
    ) -> None:
        super().__init__(chunk_size, overlap, length_function)
        if separator is None:
            raise ValueError("separator must not be None")
        self.separator = separator

    def split_text(self, text: str) -> list[Split]:
        if not text or not text.strip():
            return []  # nothing to do — never fabricate empty chunks

        # The entire class is two engine calls: CUT, then PACK.
        pieces = _split_with_offsets(text, self.separator)
        return _merge_with_overlap(
            pieces, self.separator, self.chunk_size, self.overlap, self.length_function
        )


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """Split by falling down a ladder of separators, coarsest first.

    Strategy:
        1. Cut on separator #1 (e.g. "\\n\\n" — paragraphs).
        2. Any piece STILL larger than chunk_size is re-cut with
           separator #2 ("\\n" — lines), then #3 (". " — sentences),
           then #4 (" " — words), then "" (characters, last resort).
           That re-cutting is the RECURSION in the name.
        3. Small pieces are merged greedily, with overlap, exactly like
           CharacterTextSplitter.

    Net effect: chunks respect paragraphs where possible, sentences
    where necessary, and only ever cut mid-word as a last resort.

    Use this by default for real-world documents (mixed prose, code,
    lists, tables) — it is what LangChain recommends too.

    Args:
        separators: Priority-ordered, coarse-to-fine. The FINAL entry
                    should be "" so no piece can ever defeat the ladder.
                    For source code, pass a custom ladder, e.g.
                    ["\\nfunction ", "\\ncontract ", "\\n", " ", ""].
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 80,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] = len,
    ) -> None:
        super().__init__(chunk_size, overlap, length_function)
        seps = list(separators) if separators else list(self.DEFAULT_SEPARATORS)
        if not seps:
            raise ValueError("separators must be a non-empty list")
        self.separators = seps

    def split_text(self, text: str) -> list[Split]:
        if not text or not text.strip():
            return []
        return self._recursive_split(text, base=0, separators=self.separators)

    # -- the recursion itself ------------------------------------------------ #

    def _recursive_split(self, text: str, base: int, separators: list[str]) -> list[Split]:
        """Handle `text` (which starts at `base` in the original) with one
        rung of the ladder.

        Args:
            text:       The (sub-)piece to process. May be the whole
                        document at the top call, or a too-big piece in
                        deeper calls.
            base:       Where `text` begins in the ORIGINAL document —
                        passed down so offsets stay absolute everywhere.
            separators: The remaining ladder, coarsest first.
        """
        sep = separators[0]
        remaining = separators[1:]

        # CUT. `base` makes the returned offsets absolute even though we
        # are only looking at a sub-piece of the document.
        splits = _split_with_offsets(text, sep, base=base)

        chunks: list[Split] = []
        good: list[Split] = []  # consecutive pieces small enough to merge

        for piece in splits:
            if self.length_function(piece.text) < self.chunk_size:
                # Fits the budget → queue it for greedy merging.
                good.append(piece)
            else:
                # TOO BIG for this rung. Flush the queue first — merging
                # it now (rather than at the end) preserves ORDER, so the
                # returned chunk list reads exactly like the document.
                if good:
                    chunks.extend(
                        _merge_with_overlap(
                            good, sep, self.chunk_size, self.overlap, self.length_function
                        )
                    )
                    good = []
                # Then re-cut the big piece with the NEXT (finer) rung.
                if remaining:
                    chunks.extend(
                        self._recursive_split(piece.text, base=piece.char_start, separators=remaining)
                    )
                else:
                    # Ladder exhausted. Emit the oversized piece as-is:
                    # correctness beats size — we never silently drop data.
                    chunks.append(piece)

        if good:  # tail run of small pieces
            chunks.extend(
                _merge_with_overlap(good, sep, self.chunk_size, self.overlap, self.length_function)
            )
        return chunks


# ========================================================================== #
# DEMO + SELF-CHECK                                                          #
# ========================================================================== #

if __name__ == "__main__":
    sample_text = (
        "# Orbital Handbook\n\n"
        "Orbital mechanics is the study of the motion of spacecraft and "
        "natural satellites under the influence of gravity, and it is the "
        "foundation of every mission ever flown beyond the atmosphere.\n\n"
        "## Orbits\n"
        "An orbit is the curved path that a body traces around a planet or "
        "star. A stable orbit is a perpetual free fall that keeps missing "
        "the ground, which is why astronauts float inside the station.\n\n"
        "## Delta-v\n"
        "Delta-v is the currency of spaceflight. It measures how much "
        "velocity change a spacecraft can achieve with its propellant, and "
        "mission designers spend it like a very tight budget."
    )

    def show(title: str, chunks: list[dict]) -> None:
        print("=" * 70)
        print(title)
        print("=" * 70)
        for i, c in enumerate(chunks):
            # The invariant, checked the honest way: against the source.
            assert sample_text[c["char_start"] : c["char_end"]] == c["text"]
            print(f"[{i}] chars {c['char_start']}-{c['char_end']} ({len(c['text'])} chars)")
            print("    " + c["text"].replace("\n", " / ")[:78] + "...")
        print()

    # Layer 3 in action — the API a user actually sees:
    char_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, overlap=40)
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=120, overlap=30)

    show("CharacterTextSplitter (\\n\\n, 200, overlap 40)", char_splitter.split_chunks(sample_text))
    show("RecursiveCharacterTextSplitter (120, overlap 30)", rec_splitter.split_chunks(sample_text))

    # Bonus: the SAME engine, measured in ~tokens instead of characters.
    # Swap in `tiktoken` in production — the call shape does not change.
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=25, overlap=5, length_function=lambda s: max(1, len(s) // 4)
    )
    show("RecursiveCharacterTextSplitter (~token budget)", token_splitter.split_chunks(sample_text))

    print("All chunks verified: original[char_start:char_end] == chunk['text'] ✔")