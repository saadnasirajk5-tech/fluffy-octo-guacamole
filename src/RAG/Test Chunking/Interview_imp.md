# RAG Chunking & Text Splitting — Interview Prep

Study guide built from everything we discussed: sliding-window chunking, char offsets, `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`, plus the Solidity value/reference warm-up. Grouped by how interviews actually test this: **concept Q&A**, **spot-the-bug**, and **whiteboard coding**.

---

## 1. Core concept questions (with strong answers)

### Q1. Why do RAG systems chunk documents at all?

**Answer:** Three reasons, always mention all three:
1. **Embedding quality** — an embedding of a whole document mixes many topics, so the vector's similarity signal is diluted. A chunk = one focused idea = a sharper vector.
2. **Model limits** — embedding models have max input sizes; the LLM has a finite context window.
3. **Cost & latency** — stuffing entire documents into prompts wastes tokens.

**Follow-up they love:** *What's the core tradeoff in chunk size?*
- **Small chunks** → precise retrieval, but fragments lack context and you store/scan more chunks (noise, cost).
- **Large chunks** → rich context, but diluted embeddings and risk exceeding limits.
- The real answer: **there is no magic number — you tune it per corpus and measure with retrieval metrics (recall@k, MRR), not vibes.** Saying this out loud scores seniority points.

### Q2. What is chunk overlap and why do we need it?

**Answer:** Ideas and sentences straddle chunk boundaries. Without overlap, a sentence cut in half produces two meaningless fragments and retrieval may return a partial idea. Overlap repeats the tail of chunk *N* at the head of chunk *N+1*, so any idea near a boundary appears intact in at least one chunk. Typical: **10–20% of chunk size**. Cost: more chunks, more embedding cost, some duplicated content.

### Q3. Sliding window vs. RecursiveCharacterTextSplitter — what's the difference?

**Answer:**
- **Sliding window:** cuts blindly every N characters — fast, dead simple, but boundaries can land mid-word ("...value typ|es store...").
- **Recursive splitter:** splits at **natural boundaries** using a priority ladder of separators (`"\n\n"` → `"\n"` → `". "` → `" "` → `""`), greedily merges pieces up to `chunk_size`, and **recursively re-splits** any oversized piece with the next separator. Chunks respect paragraphs → sentences → words, cutting characters only as a last resort.
- Both support overlap; the difference is *where the seams land*.

### Q4. Why should chunks record `char_start` / `char_end` offsets?

**Answer:** Offsets make chunks **traceable back to the source**:
- **Citations** — highlight exactly where a retrieved chunk came from;
- **Neighbour merging** — grab adjacent chunks when one lacks context;
- **Debugging** — verify boundaries didn't cut mid-sentence;
- **Deduplication** — detect overlapping content precisely.

State the invariant every implementation must protect:

```python
original_text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]
```

### Q5. Why do professionals chunk by tokens instead of characters?

**Answer:** Embedding models and LLMs count (and charge) in **tokens**, not characters. 500 chars of dense code can be 3× the tokens of 500 chars of prose. LangChain supports this by injecting a `length_function` (e.g., a tiktoken counter) — a textbook **Strategy pattern**. Bonus point: offsets stay character-based while *measurement* is token-based — the two units are deliberately separate concepts.

### Q6. When would you use CharacterTextSplitter over Recursive?

**Answer:** When text is uniform and dominated by **one** delimiter (CSV rows, double-newline prose). For mixed real-world docs (prose + code + lists), Recursive is the default. For source code, use a **custom ladder** — e.g. for Solidity: `["\nfunction ", "\ncontract ", "\n", " "]` so chunks never split a function.

### Q7. What data structure should a chunk be, dict or object?

**Answer:** Shows engineering maturity: a **frozen dataclass** beats a dict for production — typos fail fast at type-check time, IDEs autocomplete, instances are immutable/hashable. Keep a `.to_dict()` for notebook compatibility. For internal pieces, a `NamedTuple(text, start, end)` prevents swapped-argument bugs that plain tuples invite.

---

## 2. Spot-the-bug questions (all real bugs from our session)

Interviewers show you code and ask "what's wrong?" Practice on these — they're the exact failure modes of chunking code.

### B1. The inverted condition

```python
if not text or text.strip():
    return []
```

**Bug:** For any *valid* text, `text.strip()` is truthy → the function returns `[]` for every real document. Missing a single `not`: it must be `if not text or not text.strip():`.

### B2. Normalization breaks offsets

```python
text = text.replace("\r\n", "\n").replace("\r", "\n")
```

**Bug:** `\r\n` is 2 chars, `\n` is 1 — the string shrinks, so every `char_start`/`char_end` recorded afterwards points into the *modified* text. The invariant `original[start:end] == chunk["text"]` silently breaks for Windows documents. Fix: don't normalize, or record offsets against the normalized text and document that.

### B3. Unguarded indexing downstream

```python
chunks = chunk_text(sample_text, size, overlap)
print(chunks[0])       # 💥 IndexError on empty/whitespace-only input
```

**Bug:** Empty input → empty list → `IndexError`. Guard with `if not chunks:` before indexing, and write tests for the empty case.

### B4. The infinite-loop guard

```python
if overlap >= chunk_size:
    raise ValueError(...)
```

**Why it matters:** in a sliding window, `start` advances by `chunk_size - overlap` each iteration. If `overlap >= chunk_size`, the window never moves forward → **infinite loop**. Being able to *explain why a validation line exists* is exactly what interviewers probe.

---

## 3. Whiteboard coding tasks (with solutions)

### C1. Write a sliding-window chunker with offsets

The classic. Must-handle: validation, end-clipping, overlap advance, empty output, whitespace-only chunks.

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if not text or not text.strip():
        return []

    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        if piece.strip():                      # skip whitespace-only windows
            chunks.append({"text": piece, "char_start": start, "char_end": end})
        if end == n:
            break
        start = end - overlap                   # window slides: chunk_size - overlap > 0
    return chunks
```

**Complexity:** O(n) time, ~`n / (chunk_size - overlap)` chunks. Say that out loud.

### C2. Split on a separator while preserving offsets

The building block of every "real" splitter — `str.split()` alone can't do it.

```python
def split_with_offsets(text: str, sep: str) -> list[tuple[str, int, int]]:
    if text == "":
        return []
    if sep == "":                                        # last resort: chars
        return [(ch, i, i + 1) for i, ch in enumerate(text)]
    pieces, start = [], 0
    while True:
        idx = text.find(sep, start)
        if idx == -1:
            pieces.append((text[start:], start, len(text)))
            return pieces
        pieces.append((text[start:idx], start, idx))
        start = idx + len(sep)
```

### C3. Greedy merge with an overlap tail

The engine inside LangChain's splitters — pack pieces up to `chunk_size`; seed the next chunk with the previous chunk's tail.

```python
def greedy_merge(splits, sep, chunk_size, overlap):
    sep_len = len(sep)
    chunks, current, current_len = [], [], 0

    def assemble(cur):
        return (sep.join(p for p, _, _ in cur), cur[0][1], cur[-1][2])

    for piece, s, e in splits:
        add_len = len(piece) + (sep_len if current else 0)
        if current and current_len + add_len > chunk_size:   # would overflow → close
            chunks.append(assemble(current))
            tail, tail_len = [], 0                           # build ~overlap tail
            for prev in reversed(current):
                cost = len(prev[0]) + (sep_len if tail else 0)
                if tail_len + cost > overlap:
                    break
                tail.insert(0, prev)
                tail_len += cost
            current, current_len = tail, tail_len
            if current_len + add_len > chunk_size:           # huge piece → own chunk
                current, current_len = [], 0
        current.append((piece, s, e))
        current_len += add_len

    if current:
        chunks.append(assemble(current))
    return [c for c in chunks if c[0].strip()]
```

**Key talking point:** the overlap tail keeps **whole pieces** (sentences/words), not raw characters — that's why seams never cut sentences.

### C4. Recursive splitter skeleton (pseudocode is fine in interviews)

```text
def recursive_split(text, base, separators):
    sep, remaining = separators[0], separators[1:]
    splits = split_with_offsets(text, sep)          # offsets shifted by `base`
    out, batch = [], []
    for piece in splits:
        if len(piece) < chunk_size:
            batch.append(piece)                     # small → queue for merge
        else:
            out += merge(batch); batch = []          # flush in order
            out += recursive_split(piece, remaining)  # big → next level down
    out += merge(batch)
```

**Explain the `base` parameter:** recursive calls get *pieces* of the original, but offsets must stay absolute — shift by the piece's start instead of reslicing the whole document.

---

## 4. Solidity quickfire (from our first topic)

- **Value types** (`uint`, `int`, `bool`, `address`, fixed `bytesNN`, `enum`): store data directly; assignment **copies**.
- **Reference types** (`string`, dynamic `bytes`, arrays, `struct`, `mapping`): store a location; assignment can **alias** — must declare a data location.
- **Data locations:** `storage` (persistent on-chain), `memory` (temporary, wiped after call), `calldata` (read-only input, cheapest gas).
- Classic audit-style question: *what does this do?* `uint[] storage ref = storedArr; ref[0] = 99;` — Answer: it **writes to persistent state** through the alias. Unintended storage writes are a common real-world vulnerability class — exactly the kind of thing you'll hunt as a web3 auditor.

---

## 5. How to structure answers in the interview

1. **Lead with the tradeoff**, not the definition ("chunk size is a precision-vs-context tradeoff, tuned by measurement").
2. **State the invariant** for offsets — it signals you think about correctness contracts.
3. **Name the pattern** when you see one (Strategy = injected `length_fn`, Template Method = abstract base owning the pipeline).
4. **Mention failure modes you personally fixed** — inverted conditions, `\r\n` offset drift, infinite-loop guards. Bugs *you* hit make the best war stories.

> Review companion: the "Character & Recursive Text Splitters — from scratch" canvas has the full production-style implementation of everything in section 3.