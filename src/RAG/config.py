import os
import io
import json
import re
import time
import hashlib
from abc import ABC, abstractmethod # for defining an LLM interface
from collections import Counter
from datetime import datetime
from textwrap import dedent # clean multi-line strings
from typing import Optional # type hints

import numpy as np
from sentence_transformers import SentenceTransformer
#import faiss
#from rank_bm25 import BM250kapi
import tiktoken

# A single Config object: every single setting lives here
class Config:
  EMBEDDING_MODEL =  "all-MiniLM-L6-v2"
  CHUNK_SIZE = 500
  CHUNK_OVERLAP = 80
  TOP_K = 5  # How many relevant chunks to fetch per question?
  VECTOR_WEIGHT = 0.6
  KEYWORD_WEIGHT = 0.4

   # LLM settings.
  OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # leave "" to skip
  OPENAI_MODEL = "gpt-4o-mini"
  LLM_TEMPERATURE = 0.1          # low = more factual, less random
  LLM_PROVIDER = "openai"        # "openai" or "local"
  LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"   # Ollama default
  LOCAL_LLM_MODEL = "llama3.2"

  # Where we store generated files (database, search index)
  DATA_DIR = "./data"
  @property
  def openai_configured(self) -> bool:
    return bool(self.OPENAI_API_KEY)

# The one shared instance every cell will use.
config = Config()
os.makedirs(config.DATA_DIR, exist_ok=True)
print("Config ready.")
print(f"  Embedding model: {config.EMBEDDING_MODEL}")
print(f"  Chunk size: {config.CHUNK_SIZE} (overlap {config.CHUNK_OVERLAP})")
print(f"  Top-K: {config.TOP_K}")
print(f"  Hybrid weights: vector={config.VECTOR_WEIGHT}, keyword={config.KEYWORD_WEIGHT}")
print(f"  OpenAI configured: {config.openai_configured}")