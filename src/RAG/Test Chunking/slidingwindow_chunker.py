# Implementing a sliding window chunker 
def chunk_text(text:str, chunk_size:int=500, overlap:int=80) -> list:
      if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

      if overlap < 0:
        raise ValueError("overlap can't be negative")

      if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

      if not text or not text.strip():
        return [] 
      
      # Normalizing line endings so chunk boundaries behave consistently
      text = text.replace("\r\n", "\n").replace("\r","\n")  
      chunks =[] 
      text_len= len(text) 
      start = 0 # character index where the next chunk begins
      
      # walking through the whole text 
      while start < text_len:
        """Set the end of this chunk to either
         the start position plus the chunk_size, or the very end of the text—whichever comes first."""
        end = min(start+chunk_size, text_len) # window end, clipped at document's end
        piece = text[start:end] # slice the window out  

        # skip chunks that are only whitespace. For accuracy recording offset
        if piece.strip():  # skip windows that are pure whitespace
          chunks.append({  # Record text + true offsets
              "text":piece,
              "char_start":start,
              "char_end":end,
              }) 
          
        if end == text_len: # window reached the end - done
          break 

        # Advance the start by (chunk_size - overlap) so the next chunk begins
        # `overlap` characters before the end of the currrent chunk.
        start = end - overlap 
      return chunks

sample_text = documents_db["orbital_handbook"]["text"]
chunks = chunk_text(sample_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

print(f"Document split into {len(chunks)} chunks.")
print(f"First chunk (chars {chunks[0]['char_start']}-{chunks[0]['char_end']}): ")
print("-" * 60)
print(chunks[0]["text"][:300] + "...")
print("-" * 60)
print(f"Last chunk (chars {chunks[-1]['char_start']}-{chunks[-1]['char_end']}): ")
print(chunks[-1]["text"][:300] + "...")

# save chunks where the next cell expects them 
chunk_db = {"orbital_handbook": chunks}