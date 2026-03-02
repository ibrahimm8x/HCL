"""Language Interface: text <-> vector translation using pre-trained models."""
import torch
import numpy as np
from typing import List, Optional

from hlc.config import Config
from hlc.routing import RoutingResult


class LanguageInterface:
    """
    The LLM does NOT think. It translates.

    - Input: text -> embedding vector (via sentence-transformers)
    - Output: activated column context + query -> text (via Mistral-7B-Instruct)

    For v1, the 'translation layer' is implicit: we use the embedding
    model directly as the bridge between text and internal vectors.
    """

    def __init__(self, config: Config):
        self.config = config
        self._embedder = None
        self._tokenizer = None
        self._generator = None
        self._decoder = None

    def _load_embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model_name)

    def _load_generator(self):
        """Lazy load the language model."""
        if self._generator is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.lm_model_name,
                trust_remote_code=True,
            )
            self._generator = AutoModelForCausalLM.from_pretrained(
                self.config.lm_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def encode(self, text: str) -> np.ndarray:
        """Convert text to embedding vector (384-dim)."""
        self._load_embedder()
        embedding = self._embedder.encode([text], convert_to_numpy=True)
        return embedding[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode multiple texts."""
        self._load_embedder()
        return self._embedder.encode(texts, convert_to_numpy=True)

    def generate_response(
        self,
        context_texts: List[str],
        query_text: str,
        value_state: Optional[object] = None,
    ) -> str:
        """
        Generate text response given activated column contexts and query.

        Constructs a prompt from the knowledge retrieved by the routing
        loop and generates a coherent response using the LLM.
        """
        self._load_generator()

        # Build prompt from activated column knowledge
        if context_texts:
            context = "\n".join(f"- {t}" for t in context_texts if t)
            prompt = (
                f"You are a knowledgeable assistant. Based on the following knowledge:\n"
                f"{context}\n\n"
                f"Answer the following question concisely:\n"
                f"{query_text}\n\n"
                f"Answer:"
            )
        else:
            prompt = (
                f"You are a knowledgeable assistant. "
                f"Answer the following question concisely:\n"
                f"{query_text}\n\n"
                f"Answer:"
            )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._generator.device)

        with torch.no_grad():
            output = self._generator.generate(
                **inputs,
                max_new_tokens=self.config.lm_max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        full_text = self._tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the answer portion
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()

        return answer

    def generate_response_decoder(
        self,
        routing_result: RoutingResult,
        query_text: str,
    ) -> str:
        """
        Generate response using the fine-tuned decoder instead of LLM.

        The decoder has no knowledge of its own — it only composes
        language from the knowledge provided by the routing loop.
        """
        if self._decoder is None:
            from hlc.decoder import Decoder
            self._decoder = Decoder(self.config)

        return self._decoder.generate(routing_result, query_text)
