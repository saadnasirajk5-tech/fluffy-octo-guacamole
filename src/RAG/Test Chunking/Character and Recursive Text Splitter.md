## CharacterTextSplitter — one separator, one level

```
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, overlap=80)
```

Splits on **one** delimiter only (default: paragraph breaks). Every paragraph smaller than 500 chars gets packed with neighbors; the overlap tail carries context forward.

**Weakness:** if one paragraph is 800 chars, it can't be broken down — you just get an oversized chunk. No plan B.

## RecursiveCharacterTextSplitter — the ladder 🪜
This is why it's the most-used splitter in real RAG. It carries a **priority list of separators**:

```
separators = ["\n\n", "\n", ". ", " ", ""]
```

And the algorithm is: _split with separator #1 → any piece still too big? Recursively re-split it with separator #2 → still too big? #3..._ down to `""` (individual characters, the last resort).

```
Big text (1200 chars, chunk_size=500)
  ├─ split on "\n\n" (paragraphs)
  │    ├─ paragraph A (300)  ✓ small → merge with neighbors
  │    └─ paragraph B (800)  ✗ too big → split on "\n"  (lines)
  │          └─ line B1 (700) ✗ still too big → split on ". " (sentences)
  │                └─ sentence (150) ✓ → merge
```

The result: chunks respect paragraphs where possible, sentences where necessary, and only ever cut **mid-word as an absolute last resort**.

## When to use which?

|Situation|Use|
|---|---|
|Uniform text, one obvious delimiter (CSV rows, `\n\n` prose)|`CharacterTextSplitter`|
|Real-world mixed docs (prose + code + lists + tables)|`RecursiveCharacterTextSplitter` ← default choice|
|Code specifically|Recursive with custom ladder, e.g. `["\nclass ", "\ndef ", "\n\n", "\n", " "]`|
|Semantic precision > simplicity|Semantic chunking (embeddings-based)|

**Pro tip:** the ladder is customizable — that's a big pro trick. For Solidity source code you'd use something like `["\nfunction ", "\ncontract ", "\n", " "]` so chunks never break a function in half. That matters a lot when you build RAG over code for security work.