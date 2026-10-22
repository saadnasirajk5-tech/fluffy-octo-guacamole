# Building a BM25 keyword index 
# pip install rank-bm25


from rank_bm25 import BM25Okapi
"""it ranks documents based on how often 
your search words appear in a chunk, while downweighting words that appear everywhere."""

class keywordIndex: 
  """A BM25 keyword search index over text chunks.""" 
  def __init__(self): 
    self.bm25 : BM25Okapi | None = None 
    self.tokenized: list = [] # tokenized corpus
    self.texts: list[str] = [] # raw chunk texts
    self.id_map : list = [] # row i -> chunk index
  
  def build(self, chunks: list) -> None: # (fixed typo `-> None = None:`)
    if not chunks:
        return 
    self.texts = [c["text"] for c in chunks] 
    self.id_map = list(range(len(chunks)))
    self.tokenized = [re.findall(r"[a-z0-9]+", t.lower()) for t in self.texts] 
    self.bm25 = BM25Okapi(self.tokenized)

  def search(self, query:str, top_k:int=5) -> list:
    """Return the top-k chunks that best match the query by keywords."""
    if self.bm25 is None:
      return [] 
    # Tokenize the query in same way we tokenize the corpus 
    q_tokens = re.findall(r"[a-z0-9]+",query.lower()) 
    scores = self.bm25.get_scores(q_tokens) 
    # Find the indices of the top-k scores (argsort descending) 
    top_indices = np.argsort(scores)[::-1][:top_k] # sorts by index from lowest score to highest score.
    results = [] 
    for idx in top_indices: 
      if scores[idx] <= 0: 
        break 
      results.append({
          "chunk_index": int(idx), 
          "text":self.texts[idx],
          "score":float(scores[idx]), 
          "search_type": "keyword",
      })
    return results
# --- Build the BM25 index over our sample document's chunks. ---
keyword_index =  keywordIndex()
keyword_index.build(chunk_db["orbital_handbook"])
print(f"Keyword index built: {len(keyword_index.texts)} chunks indexed.")

# --- Quick demo: search for a question with an exact term. ---
demo_query = "What encryption standard does Orbital use?"
demo_results = keyword_index.search(demo_query, top_k=3)
print(f"")
print(f"Query: {demo_query}")
print(f"Top {len(demo_results)} keyword results:")
for i, r in enumerate(demo_results, 1):
    print(f"  {i}. [score {r['score']:.2f}] {r['text'][:120]}...")