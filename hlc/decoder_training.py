"""Decoder Training: generate data via local Mistral-7B, fine-tune GPT-2 with LoRA."""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from hlc.config import Config
from hlc.decoder import ROUTING_TOKEN, RESPONSE_TOKEN, END_TOKEN


@dataclass
class TrainingExample:
    """A single (prompt, response) training pair."""
    query: str
    knowledge: List[str]        # active_source_texts from routing
    value_joy: float
    value_curiosity: float
    value_pain: float
    confidence: str             # "high", "medium", "low"
    mode: str                   # "fast", "light", "slow"
    response: str               # target text the decoder should learn to produce

    def to_training_text(self) -> str:
        """Format as the full training sequence the decoder learns."""
        knowledge_lines = "\n".join(f"- {k}" for k in self.knowledge) if self.knowledge else "- No relevant knowledge found"
        state = (
            f"joy={self.value_joy:.2f} curiosity={self.value_curiosity:.2f} "
            f"pain={self.value_pain:.2f} confidence={self.confidence} mode={self.mode}"
        )
        return (
            f"{ROUTING_TOKEN}\n"
            f"QUERY: {self.query}\n"
            f"KNOWLEDGE:\n{knowledge_lines}\n"
            f"STATE: {state}\n"
            f"{RESPONSE_TOKEN}\n"
            f"{self.response}\n"
            f"{END_TOKEN}"
        )


class TeacherLLM:
    """
    Local open-source LLM used to generate decoder training data.
    Uses the same Mistral-7B-Instruct already in our config.
    Runs on Colab GPU — zero API cost.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy load Mistral-7B-Instruct."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading teacher model: {self.config.lm_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.lm_model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.lm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("Teacher model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Generate text from a prompt using the local LLM."""
        import torch

        self._load()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        full_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        # Strip the prompt from the output
        response = full_text[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        return response


class DecoderTrainer:
    """
    Generates training data using a local open-source LLM (Mistral-7B)
    and fine-tunes GPT-2 Medium with LoRA.

    Pipeline:
    1. generate_questions() — use Mistral-7B to create questions for each fact
    2. generate_responses() — use Mistral-7B to create target responses
    3. augment_dataset() — expand base pairs to 10K with variations
    4. train() — LoRA fine-tune GPT-2 Medium
    """

    def __init__(self, config: Config):
        self.config = config
        self._teacher = None

    def _get_teacher(self) -> TeacherLLM:
        """Get or create the teacher LLM."""
        if self._teacher is None:
            self._teacher = TeacherLLM(self.config)
        return self._teacher

    def generate_questions(self, facts: List[str]) -> List[Dict]:
        """
        Generate questions for each fact via Mistral-7B.

        Returns list of {fact, questions: [str]} dicts.
        """
        teacher = self._get_teacher()
        results = []

        for i, fact in enumerate(facts):
            if fact.startswith("STRATEGY:"):
                continue  # Skip strategy columns for now

            prompt = (
                f"[INST] Given this fact:\n\"{fact}\"\n\n"
                f"Generate exactly 5 questions that can be answered using this fact.\n"
                f"Output ONLY the questions, one per line, numbered 1-5.\n"
                f"Include 2 direct questions, 2 paraphrased questions, and "
                f"1 question where this fact alone is not enough to fully answer. [/INST]\n"
            )

            try:
                raw = teacher.generate(prompt, max_new_tokens=300)
                # Parse numbered questions from response
                questions = self._parse_questions(raw)
                if len(questions) >= 2:
                    results.append({"fact": fact, "questions": questions[:5]})
                else:
                    raise ValueError(f"Only parsed {len(questions)} questions")
            except Exception as e:
                print(f"  Warning: failed for fact {i}: {e}")
                # Fallback: template-based questions
                results.append({
                    "fact": fact,
                    "questions": self._template_questions(fact),
                })

            if (i + 1) % 10 == 0:
                print(f"  Generated questions for {i + 1}/{len(facts)} facts...")

        return results

    def _parse_questions(self, raw_text: str) -> List[str]:
        """Parse numbered questions from LLM output."""
        questions = []
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering: "1. ", "1) ", "- ", etc.
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            if cleaned and "?" in cleaned:
                questions.append(cleaned)
        return questions

    def _template_questions(self, fact: str) -> List[str]:
        """Generate simple template-based questions as fallback."""
        # Extract key terms for question generation
        words = fact.split()
        short = " ".join(words[:8])
        return [
            f"What can you tell me about {short}?",
            f"Explain: {short}.",
            f"How does this work: {short}?",
        ]

    def generate_responses(
        self,
        query: str,
        knowledge_texts: List[str],
        value_joy: float = 0.5,
        value_curiosity: float = 0.1,
        value_pain: float = 0.0,
    ) -> str:
        """
        Use Mistral-7B to generate a target response given routing output.

        Critical constraint: response must ONLY use the provided knowledge.
        """
        teacher = self._get_teacher()

        if knowledge_texts:
            knowledge_str = "\n".join(f"- {k}" for k in knowledge_texts)
        else:
            knowledge_str = "(none)"

        # Describe the tone based on value state
        if value_joy > 0.6:
            tone = "Answer confidently and directly."
        elif value_curiosity > 0.4:
            tone = "Answer with an exploratory, curious tone."
        elif value_pain > 0.3:
            tone = "Answer cautiously, expressing uncertainty."
        else:
            tone = "Answer in a neutral, informative tone."

        prompt = (
            f"[INST] You are a concise assistant. You must ONLY use the knowledge provided below.\n"
            f"Do NOT add any facts not listed. If the knowledge is insufficient, say "
            f"\"I don't have enough information about that.\"\n"
            f"Keep your response to 1-3 sentences. {tone}\n\n"
            f"Available knowledge:\n{knowledge_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely using ONLY the knowledge above: [/INST]\n"
        )

        response = teacher.generate(prompt, max_new_tokens=100)

        # Clean up: take only the first 1-3 sentences
        sentences = response.split(".")
        cleaned = ".".join(sentences[:3]).strip()
        if cleaned and not cleaned.endswith("."):
            cleaned += "."
        return cleaned if cleaned and len(cleaned) > 5 else response.strip()[:200]

    def generate_base_dataset(
        self,
        facts: List[str],
        output_path: Path,
        verbose: bool = True,
    ) -> List[TrainingExample]:
        """
        Generate the base training dataset using Mistral-7B.

        For each fact, generates questions and target responses.
        Saves to JSONL file. Runs entirely on local/Colab GPU — no API cost.
        """
        if verbose:
            print("Step 1: Generating questions for each fact...")
        fact_questions = self.generate_questions(facts)

        if verbose:
            total_q = sum(len(fq["questions"]) for fq in fact_questions)
            print(f"  Generated {total_q} questions from {len(fact_questions)} facts.")
            print("Step 2: Generating target responses...")

        examples = []
        count = 0
        for fq in fact_questions:
            fact = fq["fact"]
            for question in fq["questions"]:
                # High confidence scenario: just this fact
                response = self.generate_responses(
                    question, [fact], value_joy=0.8, value_curiosity=0.1,
                )
                examples.append(TrainingExample(
                    query=question,
                    knowledge=[fact],
                    value_joy=0.8,
                    value_curiosity=0.1,
                    value_pain=0.0,
                    confidence="high",
                    mode="fast",
                    response=response,
                ))

                count += 1
                if verbose and count % 20 == 0:
                    print(f"  Generated {count} training pairs...")

        # Add "no knowledge" examples
        no_knowledge_queries = [
            "What is the meaning of life?",
            "Who won the world series in 2024?",
            "What is the best programming language?",
            "How do I bake a chocolate cake?",
            "What will happen tomorrow?",
            "Tell me about quantum computing.",
            "What color is happiness?",
            "How many stars are in the sky?",
        ]
        for q in no_knowledge_queries:
            response = self.generate_responses(
                q, [], value_joy=0.0, value_curiosity=0.6, value_pain=0.2,
            )
            examples.append(TrainingExample(
                query=q,
                knowledge=[],
                value_joy=0.0,
                value_curiosity=0.6,
                value_pain=0.2,
                confidence="low",
                mode="slow",
                response=response,
            ))

        # Save base dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")

        if verbose:
            print(f"  Saved {len(examples)} base examples to {output_path}")

        return examples

    def augment_dataset(
        self,
        base_examples: List[TrainingExample],
        target_size: int = 10000,
    ) -> List[TrainingExample]:
        """
        Augment base examples to target size with variations.

        Variations:
        - Shuffle knowledge bullet order
        - Vary value states (same query, different tone)
        - Vary number of knowledge bullets (add related noise)
        - Change routing mode
        """
        augmented = list(base_examples)

        while len(augmented) < target_size:
            base = random.choice(base_examples)

            variation_type = random.choice([
                "shuffle_knowledge",
                "vary_values",
                "add_noise_knowledge",
                "change_mode",
            ])

            if variation_type == "shuffle_knowledge" and len(base.knowledge) > 1:
                new = TrainingExample(
                    query=base.query,
                    knowledge=random.sample(base.knowledge, len(base.knowledge)),
                    value_joy=base.value_joy,
                    value_curiosity=base.value_curiosity,
                    value_pain=base.value_pain,
                    confidence=base.confidence,
                    mode=base.mode,
                    response=base.response,
                )
                augmented.append(new)

            elif variation_type == "vary_values":
                joy = random.uniform(0.0, 1.0)
                curiosity = random.uniform(0.0, 0.6)
                pain = random.uniform(0.0, 0.3)
                conf = "high" if joy > 0.6 else ("medium" if joy > 0.3 else "low")
                new = TrainingExample(
                    query=base.query,
                    knowledge=base.knowledge,
                    value_joy=joy,
                    value_curiosity=curiosity,
                    value_pain=pain,
                    confidence=conf,
                    mode=base.mode,
                    response=base.response,
                )
                augmented.append(new)

            elif variation_type == "add_noise_knowledge":
                other = random.choice(base_examples)
                if other.knowledge and other.knowledge != base.knowledge:
                    combined = base.knowledge + [random.choice(other.knowledge)]
                    random.shuffle(combined)
                    new = TrainingExample(
                        query=base.query,
                        knowledge=combined[:7],  # Cap at WM capacity
                        value_joy=base.value_joy,
                        value_curiosity=base.value_curiosity,
                        value_pain=base.value_pain,
                        confidence=base.confidence,
                        mode=base.mode,
                        response=base.response,
                    )
                    augmented.append(new)

            elif variation_type == "change_mode":
                new_mode = random.choice(["fast", "light", "slow"])
                new = TrainingExample(
                    query=base.query,
                    knowledge=base.knowledge,
                    value_joy=base.value_joy,
                    value_curiosity=base.value_curiosity,
                    value_pain=base.value_pain,
                    confidence=base.confidence,
                    mode=new_mode,
                    response=base.response,
                )
                augmented.append(new)

        return augmented[:target_size]

    def prepare_training_data(self, examples: List[TrainingExample], output_path: Path):
        """Convert examples to the text format the decoder trains on."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                text = ex.to_training_text()
                f.write(json.dumps({"text": text}) + "\n")
        print(f"Saved {len(examples)} training texts to {output_path}")

    def train(
        self,
        training_data_path: Path,
        output_adapter_path: Optional[Path] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        """
        Fine-tune GPT-2 Medium with LoRA on the training data.

        Designed to run on Colab T4 (~2 hours for 10K examples).
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset

        output_adapter_path = output_adapter_path or self.config.decoder_adapter_path
        device = self.config.get_device()

        print(f"Loading base model: {self.config.decoder_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_model_name)

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [ROUTING_TOKEN, RESPONSE_TOKEN, END_TOKEN],
        }
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = tokenizer.eos_token
        tokenizer.add_special_tokens(special_tokens)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name,
            torch_dtype=torch.float32,
        )
        model.resize_token_embeddings(len(tokenizer))

        # Configure LoRA
        print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load and tokenize dataset
        print(f"Loading training data from {training_data_path}")
        dataset = load_dataset("json", data_files=str(training_data_path), split="train")

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # For causal LM, labels = input_ids
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        tokenized = tokenized.map(add_labels, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_adapter_path.parent / "decoder_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
        )

        print("Starting LoRA fine-tuning...")
        trainer.train()

        # Save LoRA adapter
        print(f"Saving adapter to {output_adapter_path}")
        output_adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_adapter_path))
        tokenizer.save_pretrained(str(output_adapter_path))

        print("Training complete.")
