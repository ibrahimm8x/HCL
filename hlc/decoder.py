"""Decoder: MiniT5 encoder-decoder trained from scratch. Zero pretrained knowledge."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from hlc.config import Config
from hlc.routing import RoutingResult


# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
END_TOKEN = "<end>"
QUERY_TOKEN = "<query>"
KNOWLEDGE_TOKEN = "<knowledge>"
STATE_TOKEN = "<state>"
NOKNOWLEDGE_TOKEN = "<noknowledge>"


@dataclass
class MiniT5Config:
    """Architecture config for the MiniT5 decoder model."""
    vocab_size: int = 50257 + 7  # GPT-2 tokenizer + 7 special tokens
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_token_id: int = 50257     # first special token


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MiniT5(nn.Module):
    """
    Small T5-style encoder-decoder. ~20M parameters.
    Trained from scratch — zero pretrained knowledge.

    Encoder reads: query + knowledge bullets + value state
    Decoder writes: natural language response
    """

    def __init__(self, config: MiniT5Config):
        super().__init__()
        self.config = config

        # Shared embedding for encoder and decoder
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        # Output projection (tied with embedding weights)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # weight tying

        # Pointer-generator: copy words directly from encoder input
        self.copy_attn = nn.MultiheadAttention(
            config.d_model, num_heads=1, batch_first=True,
        )
        self.p_gen_gate = nn.Linear(config.d_model * 2, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1 and not p.is_shared():
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def encode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source sequence. Returns memory: (batch, src_len, d_model)."""
        src_emb = self.pos_encoder(self.embedding(src_ids))
        return self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence. Returns decoder output: (batch, tgt_len, d_model)."""
        tgt_emb = self.pos_encoder(self.embedding(tgt_ids))
        tgt_mask = self._make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        return self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

    def pointer_generator(
        self,
        decoder_output: torch.Tensor,  # (batch, tgt_len, d_model)
        memory: torch.Tensor,           # (batch, src_len, d_model)
        src_ids: torch.Tensor,           # (batch, src_len)
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pointer-generator: mix vocab generation with copying from source.

        At each step, the model can either:
        1. Generate a word from the full vocabulary (standard path)
        2. Copy a word directly from the encoder input (pointer path)

        A learned gate p_gen decides the mix. This lets the model output
        ANY word from the knowledge bullets, even words never seen in training.

        Returns: final_dist (batch, tgt_len, vocab_size) — probabilities
        """
        # Standard vocab distribution
        vocab_logits = self.output_proj(decoder_output)
        vocab_dist = F.softmax(vocab_logits, dim=-1)

        # Copy attention: decoder attends to encoder to decide what to copy
        context, copy_attn_weights = self.copy_attn(
            decoder_output, memory, memory,
            key_padding_mask=src_pad_mask,
        )
        # copy_attn_weights: (batch, tgt_len, src_len) — attention over source tokens

        # Gate: p_gen = probability of generating from vocab (vs copying)
        p_gen = torch.sigmoid(self.p_gen_gate(
            torch.cat([decoder_output, context], dim=-1)
        ))
        # p_gen: (batch, tgt_len, 1)

        # Copy distribution: scatter source attention into vocab space
        batch_size, tgt_len, src_len = copy_attn_weights.shape
        vocab_size = vocab_logits.size(-1)

        copy_dist = torch.zeros(batch_size, tgt_len, vocab_size, device=src_ids.device)
        src_ids_expanded = src_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        copy_dist.scatter_add_(2, src_ids_expanded, copy_attn_weights)

        # Mix: generate from vocab OR copy from source
        final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist

        # Epsilon for numerical stability (avoid log(0))
        final_dist = final_dist + 1e-10

        return final_dist

    def forward(
        self,
        src_ids: torch.Tensor,       # (batch, src_len)
        tgt_ids: torch.Tensor,       # (batch, tgt_len)
        src_pad_mask: Optional[torch.Tensor] = None,  # (batch, src_len)
        tgt_pad_mask: Optional[torch.Tensor] = None,  # (batch, tgt_len)
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Returns: final_dist (batch, tgt_len, vocab_size) — probabilities
        Uses pointer-generator to mix vocab generation with source copying.
        """
        memory = self.encode(src_ids, src_pad_mask)
        decoder_output = self.decode(tgt_ids, memory, src_pad_mask, tgt_pad_mask)
        return self.pointer_generator(decoder_output, memory, src_ids, src_pad_mask)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Decoder:
    """
    The mouth. Translates routing output into natural language.

    Uses MiniT5 trained from scratch — zero pretrained knowledge.
    All facts come from columns via RoutingResult.active_source_texts.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model: Optional[MiniT5] = None
        self._tokenizer = None
        self._special_token_ids = {}

    def _load(self):
        """Load the trained MiniT5 model."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer

        # Load GPT-2 tokenizer and add our special tokens
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        special_tokens = {
            "additional_special_tokens": [
                PAD_TOKEN, BOS_TOKEN, END_TOKEN,
                QUERY_TOKEN, KNOWLEDGE_TOKEN, STATE_TOKEN, NOKNOWLEDGE_TOKEN,
            ],
        }
        self._tokenizer.add_special_tokens(special_tokens)
        self._tokenizer.pad_token = PAD_TOKEN

        # Cache special token IDs
        for tok in [PAD_TOKEN, BOS_TOKEN, END_TOKEN, QUERY_TOKEN, KNOWLEDGE_TOKEN, STATE_TOKEN, NOKNOWLEDGE_TOKEN]:
            self._special_token_ids[tok] = self._tokenizer.convert_tokens_to_ids(tok)

        # Build model
        model_config = MiniT5Config(
            vocab_size=len(self._tokenizer),
            d_model=self.config.decoder_d_model,
            n_heads=self.config.decoder_n_heads,
            n_encoder_layers=self.config.decoder_n_encoder_layers,
            n_decoder_layers=self.config.decoder_n_decoder_layers,
            d_ff=self.config.decoder_d_ff,
            max_seq_len=self.config.decoder_max_seq_len,
            dropout=self.config.decoder_dropout,
            pad_token_id=self._special_token_ids[PAD_TOKEN],
        )
        self._model = MiniT5(model_config)

        # Load trained weights if they exist
        model_path = self.config.decoder_model_path
        weights_file = model_path / "model.pt"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=self.config.get_device())
            self._model.load_state_dict(state_dict)

        self._model.to(self.config.get_device())
        self._model.eval()

    def format_encoder_input(self, routing_result: RoutingResult, query: str) -> str:
        """
        Build the encoder input string from routing output.

        Format: <query> What is DNA? <knowledge> DNA contains... <state> joy=0.80 mode=fast
        """
        parts = [f"{QUERY_TOKEN} {query}"]

        if routing_result.active_source_texts:
            knowledge = " ".join(routing_result.active_source_texts)
            parts.append(f"{KNOWLEDGE_TOKEN} {knowledge}")
        else:
            parts.append(NOKNOWLEDGE_TOKEN)

        vs = routing_result.value_state
        confidence = "high" if routing_result.prediction_error < 0.2 else (
            "medium" if routing_result.prediction_error < 0.5 else "low"
        )
        state = f"joy={vs.joy:.2f} curiosity={vs.curiosity:.2f} pain={vs.pain:.2f} confidence={confidence} mode={routing_result.mode}"
        parts.append(f"{STATE_TOKEN} {state}")

        return " ".join(parts)

    def generate(self, routing_result: RoutingResult, query: str) -> str:
        """
        Generate a text response from routing output.
        Autoregressive decoding with early stopping at <end>.
        """
        self._load()
        device = self.config.get_device()

        # Encode input
        encoder_text = self.format_encoder_input(routing_result, query)
        src_tokens = self._tokenizer(
            encoder_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.decoder_max_seq_len,
            padding=False,
        )
        src_ids = src_tokens["input_ids"].to(device)

        # Start decoder with <bos> + first knowledge token (if available)
        # The model struggles to predict the very first word after <bos>
        # because it has zero decoder context. Seeding with the first token
        # from knowledge gives it a head start — it copies perfectly from
        # position 2 onward.
        bos_id = self._special_token_ids[BOS_TOKEN]
        end_id = self._special_token_ids[END_TOKEN]

        if routing_result.active_source_texts:
            knowledge_text = " ".join(routing_result.active_source_texts)
            knowledge_tokens = self._tokenizer(
                knowledge_text, add_special_tokens=False,
            )["input_ids"]
            # Seed with first token from knowledge
            seed = knowledge_tokens[:1]
            tgt_ids = torch.tensor([[bos_id] + seed], device=device)
        else:
            tgt_ids = torch.tensor([[bos_id]], device=device)

        # Autoregressive generation with pointer-generator
        with torch.no_grad():
            # Encode once
            memory = self._model.encode(src_ids)

            for _ in range(self.config.decoder_max_output_tokens):
                decoder_output = self._model.decode(tgt_ids, memory)

                # Pointer-generator: can copy words from encoder input
                final_dist = self._model.pointer_generator(
                    decoder_output[:, -1:, :],  # last position only
                    memory, src_ids,
                )

                # Greedy selection from combined distribution
                next_token = final_dist[:, 0, :].argmax(dim=-1, keepdim=True)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

                if next_token.item() == end_id:
                    break

        # Decode tokens to text (skip <bos>, stop before <end>)
        output_ids = tgt_ids[0, 1:].tolist()  # skip <bos>
        if end_id in output_ids:
            output_ids = output_ids[:output_ids.index(end_id)]

        response = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return response

    def save(self, path=None):
        """Save model weights."""
        path = path or self.config.decoder_model_path
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            torch.save(self._model.state_dict(), path / "model.pt")
            self._tokenizer.save_pretrained(str(path))

    @staticmethod
    def get_tokenizer():
        """Get the tokenizer with special tokens (for training pipeline)."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        special_tokens = {
            "additional_special_tokens": [
                PAD_TOKEN, BOS_TOKEN, END_TOKEN,
                QUERY_TOKEN, KNOWLEDGE_TOKEN, STATE_TOKEN, NOKNOWLEDGE_TOKEN,
            ],
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token = PAD_TOKEN
        return tokenizer
