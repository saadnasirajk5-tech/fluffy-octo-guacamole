# Build a FAISS vector index from our chunks 
import faiss
"""1) Encode every chunk's text into a vector.
2) Create a FAISS inner-product index and add all vectors.
3) keep a parallel list(id_map) so we can go from a FAISS row index
   back to the actual chunk dict.
 """
class VectorIndex: # Groups our search engine logic into a reusable object.
  """A FAISS based vector search index over text chunks""" 
  def __init__(self): 
    self.index: faiss.IndexFlatIP | None = None 
    self.id_map: list =[] # Parallel list keeping track of row mappings. FAISS row i-> chunk at id_map[i] 
    self.texts: list[str] = [] # FAISS row i -> chunk text

  def build(self, chunks:list, embedder:EmbeddingService) -> None:
    """Encode all chunks and build FAISS index from them.""" 
    if not chunks:
      return 
    self.texts = [c["text"] for c in chunks] 
    vectors = embedder.encode(self.texts) 
    # Create the index. IndexFlatIP= Exact search via inner product 
    dim = vectors.shape[1] 
    self.index = faiss.IndexFlatIP(dim) 
    self.index.add(vectors) # add all vectors at once 
    # The id_map lets us go from FAISS row index back to the chunk
    self.id_map = list(range(len(chunks)))

  def search(self, query:str, embedder:EmbeddingService, top_k:int=5):
    """Return the top_k chunks most similiar to `query`."""
    # Encode the query into a vector (reshape to 1 x dim for FAISS)
    q_vec = embedder.encode_one(query).reshape(1, -1) # FAISS expects a 2d array even if you search only for one query
    """Asking FAISS to find the closest vectors. It returns how similar each score is. 
      Position: The row index of matching chunks in the index 
      min(..) ensures we dont't ask for more results than chunks we actually have. 
    """
    scores, positions = self.index.search(q_vec, min(top_k, self.index.ntotal)) 
    results= [] 
    for score, pos in zip(scores[0], positions[0]):
      if pos < 0: 
        continue 
      results.append({
          "chunk_index": int(pos), 
          "text":self.texts[pos], 
          "score":float(score), 
          "search_type": "vector",}) 
    return results

# --- Build the index over our sample document's chunks. ---
vector_index = VectorIndex()
if embedding_service is not None:
  vector_index.build(chunk_db["orbital_handbook"], embedding_service) 
  print(f"Vector index built: {vector_index.index.ntotal} chunks indexed.") 
  demo_query = "How much does the Orbital Link terminal cost?"
  demo_results = vector_index.search(demo_query, embedding_service, top_k=3)
  print(f"")
  print(f"Query: {demo_query}")
  print(f"Top {len(demo_results)} vector results:")
  for i, r in enumerate(demo_results, 1):
      print(f"  {i}. [score {r['score']:.3f}] {r['text'][:120]}...")
else:
    print("SKIPPED (embedding service not available).")   