"""Decoder: translates routing output into natural language. No knowledge of its own."""
import torch
from typing import Optional

from hlc.config import Config
from hlc.routing import RoutingResult


# Special tokens for the structured prompt format
ROUTING_TOKEN = "<|routing|>"
RESPONSE_TOKEN = "<|response|>"
END_TOKEN = "<|end|>"


class Decoder:
    """
    The mouth. Translates structured routing output into natural language.

    All knowledge comes from columns (via RoutingResult.active_source_texts).
    The decoder only learns HOW to say things, not WHAT to say.

    Base model: GPT-2 Medium (355M) fine-tuned with LoRA.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy load the decoder model + LoRA adapter."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.decoder_model_name,
        )

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [ROUTING_TOKEN, RESPONSE_TOKEN, END_TOKEN],
        }
        if self._tokenizer.pad_token is None:
            special_tokens["pad_token"] = self._tokenizer.eos_token
        self._tokenizer.add_special_tokens(special_tokens)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name,
            torch_dtype=torch.float32,  # GPT-2 Medium is small enough for fp32
        )
        self._model.resize_token_embeddings(len(self._tokenizer))

        # Load LoRA adapter if it exists
        adapter_path = self.config.decoder_adapter_path
        if adapter_path.exists():
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(adapter_path))

        self._model.to(self.config.get_device())
        self._model.eval()

    def format_prompt(self, routing_result: RoutingResult, query: str) -> str:
        """
        Build the structured text prompt from routing output.

        Format:
            <|routing|>
            QUERY: ...
            KNOWLEDGE:
            - fact 1
            - fact 2
            STATE: joy=0.82 curiosity=0.05 confidence=high mode=fast
            <|response|>
        """
        # Build knowledge section
        knowledge_lines = []
        for text in routing_result.active_source_texts:
            if text:
                knowledge_lines.append(f"- {text}")

        if knowledge_lines:
            knowledge = "\n".join(knowledge_lines)
        else:
            knowledge = "- No relevant knowledge found"

        # Build state section
        vs = routing_result.value_state
        confidence = "high" if routing_result.prediction_error < 0.2 else (
            "medium" if routing_result.prediction_error < 0.5 else "low"
        )
        state = (
            f"joy={vs.joy:.2f} curiosity={vs.curiosity:.2f} "
            f"pain={vs.pain:.2f} confidence={confidence} mode={routing_result.mode}"
        )

        prompt = (
            f"{ROUTING_TOKEN}\n"
            f"QUERY: {query}\n"
            f"KNOWLEDGE:\n{knowledge}\n"
            f"STATE: {state}\n"
            f"{RESPONSE_TOKEN}\n"
        )
        return prompt

    def generate(self, routing_result: RoutingResult, query: str) -> str:
        """
        Generate a text response from routing output.

        The decoder composes language from the knowledge bullets
        provided by the routing loop. It adds no facts of its own.
        """
        self._load()

        prompt = self.format_prompt(routing_result, query)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._model.device)

        # Stop generation at <|end|> or <|endoftext|>
        end_token_id = self._tokenizer.convert_tokens_to_ids(END_TOKEN)
        eos_token_id = self._tokenizer.eos_token_id
        stop_ids = [t for t in [end_token_id, eos_token_id] if t is not None]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.decoder_max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=stop_ids,
            )

        full_text = self._tokenizer.decode(output[0], skip_special_tokens=False)

        # Extract text between <|response|> and <|end|> or <|endoftext|>
        response = self._extract_response(full_text)
        return response

    def _extract_response(self, full_text: str) -> str:
        """Extract the response portion from generated text."""
        if RESPONSE_TOKEN in full_text:
            after_response = full_text.split(RESPONSE_TOKEN)[-1]
        else:
            after_response = full_text

        # Cut at whichever stop marker comes first
        for stop in [END_TOKEN, "<|endoftext|>"]:
            if stop in after_response:
                after_response = after_response.split(stop)[0]
                break

        # Take only the first 1-3 sentences to prevent runaway output
        text = after_response.strip()
        sentences = text.split(".")
        if len(sentences) > 3:
            text = ".".join(sentences[:3]).strip() + "."

        return text
