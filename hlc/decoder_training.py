"""Decoder Training: generate data via Mistral-7B, train MiniT5 from scratch."""
import json
import random
import re
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from hlc.config import Config
from hlc.decoder import (
    MiniT5, MiniT5Config, Decoder,
    PAD_TOKEN, BOS_TOKEN, END_TOKEN,
    QUERY_TOKEN, KNOWLEDGE_TOKEN, STATE_TOKEN, NOKNOWLEDGE_TOKEN,
)


NO_KNOWLEDGE_RESPONSES = [
    "I don't have enough information about that.",
    "I don't have information on that topic.",
    "I'm not able to answer that with my current knowledge.",
    "That's outside what I currently know about.",
    "I don't have relevant knowledge to answer that question.",
]

FALSE_FACTS = [
    ("What is the sun made of?", "The sun is made of cheese."),
    ("How many legs do cats have?", "Cats have twelve legs."),
    ("Which direction does water flow?", "Water naturally flows upward."),
    ("What shape is the Earth?", "The Earth is shaped like a cube."),
    ("What color is the sky?", "The sky is bright green."),
    ("How many moons does Earth have?", "Earth has seven moons."),
    ("What do fish breathe?", "Fish breathe fire."),
    ("What is ice made of?", "Ice is made of sand."),
    ("How fast does sound travel?", "Sound travels faster than light."),
    ("What do plants eat?", "Plants eat small rocks for energy."),
    ("What is the largest ocean?", "The largest ocean is made of orange juice."),
    ("How many bones do humans have?", "Humans have exactly three bones."),
    ("What causes rain?", "Rain is caused by clouds sneezing."),
    ("What is gravity?", "Gravity is a force that pushes objects apart."),
    ("What do bees produce?", "Bees produce tiny diamonds."),
    ("What is the boiling point of iron?", "Iron boils at 5 degrees Celsius."),
    ("How do birds fly?", "Birds fly by spinning their tails like propellers."),
    ("What is the capital of the Moon?", "The capital of the Moon is Lunaville."),
    ("What are clouds made of?", "Clouds are made of cotton candy."),
    ("How long is a day on Jupiter?", "A day on Jupiter lasts exactly 1 minute."),
    ("What metal is the lightest?", "Gold is the lightest metal known to science."),
    ("What is the speed of dark?", "Dark travels at exactly 42 kilometers per hour."),
    ("How deep is the Mariana Trench?", "The Mariana Trench is 3 centimeters deep."),
    ("What language do computers speak?", "Computers speak ancient Latin natively."),
    ("What temperature is absolute zero?", "Absolute zero is 500 degrees Celsius."),
    ("What is the smallest planet?", "The smallest planet is Earth."),
    ("How old is the universe?", "The universe is exactly 47 years old."),
    ("What is DNA made of?", "DNA is made of tiny magnets."),
    ("How many elements are there?", "There are exactly four elements: earth, water, fire, and air."),
    ("What do volcanoes produce?", "Volcanoes produce fresh chocolate lava."),
    ("How fast does the Earth spin?", "The Earth does not spin at all."),
    ("What is the hardest substance?", "Paper is the hardest substance known."),
    ("What causes earthquakes?", "Earthquakes are caused by giant underground worms."),
    ("What is the tallest mountain?", "The tallest mountain is 12 meters high."),
    ("What do stars produce?", "Stars produce cold darkness."),
    ("What is electricity?", "Electricity is the flow of tiny invisible hamsters."),
    ("How many continents are there?", "There are exactly two continents on Earth."),
    ("What is the freezing point of mercury?", "Mercury freezes at 200 degrees Celsius."),
    ("What are atoms made of?", "Atoms are made of tiny musical notes."),
    ("How long does it take light to reach Earth from the Sun?", "Light from the Sun takes 47 years to reach Earth."),
]


@dataclass
class TrainingExample:
    """A single (encoder_input, decoder_target) training pair."""
    query: str
    knowledge: List[str]
    value_joy: float
    value_curiosity: float
    value_pain: float
    confidence: str
    mode: str
    response: str

    def to_encoder_input(self) -> str:
        """Format as encoder input string."""
        parts = [f"{QUERY_TOKEN} {self.query}"]
        if self.knowledge:
            knowledge = " ".join(self.knowledge)
            parts.append(f"{KNOWLEDGE_TOKEN} {knowledge}")
        else:
            parts.append(NOKNOWLEDGE_TOKEN)
        state = (
            f"joy={self.value_joy:.2f} curiosity={self.value_curiosity:.2f} "
            f"pain={self.value_pain:.2f} confidence={self.confidence} mode={self.mode}"
        )
        parts.append(f"{STATE_TOKEN} {state}")
        return " ".join(parts)

    def to_decoder_target(self) -> str:
        """Format as decoder target string (with <bos> and <end>)."""
        return f"{BOS_TOKEN} {self.response} {END_TOKEN}"


class TeacherLLM:
    """
    Local Mistral-7B used to generate training data.
    Runs on Colab GPU — zero API cost.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading teacher model: {self.config.lm_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.lm_model_name, trust_remote_code=True,
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
        import torch

        self._load()
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
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
        response = full_text[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        return response


class DecoderTrainer:
    """
    Generates training data using Mistral-7B and trains MiniT5 from scratch.

    Pipeline:
    1. generate_questions() — 10 questions per fact
    2. generate_responses() — target responses from teacher
    3. generate_cross_domain_pairs() — multi-fact questions
    4. generate_no_knowledge_pairs() — "I don't know" examples
    5. augment_dataset() — expand to 60K
    6. train() — seq2seq from scratch
    """

    def __init__(self, config: Config):
        self.config = config
        self._teacher = None

    def _get_teacher(self) -> TeacherLLM:
        if self._teacher is None:
            self._teacher = TeacherLLM(self.config)
        return self._teacher

    def generate_questions(self, facts: List[str]) -> List[Dict]:
        """Generate 10 questions per fact via Mistral-7B."""
        teacher = self._get_teacher()
        results = []

        for i, fact in enumerate(facts):
            if fact.startswith("STRATEGY:"):
                continue

            prompt = (
                f"[INST] Given this fact:\n\"{fact}\"\n\n"
                f"Generate exactly 10 diverse questions that can be answered using this fact.\n"
                f"Include direct questions, paraphrased questions, and questions from different angles.\n"
                f"Output ONLY the questions, one per line, numbered 1-10. [/INST]\n"
            )

            try:
                raw = teacher.generate(prompt, max_new_tokens=400)
                questions = self._parse_questions(raw)
                if len(questions) >= 3:
                    results.append({"fact": fact, "questions": questions[:10]})
                else:
                    raise ValueError(f"Only parsed {len(questions)} questions")
            except Exception as e:
                print(f"  Warning: fact {i}: {e}")
                results.append({"fact": fact, "questions": self._template_questions(fact)})

            if (i + 1) % 10 == 0:
                print(f"  Generated questions for {i + 1}/{len(facts)} facts...")

        return results

    def _parse_questions(self, raw_text: str) -> List[str]:
        questions = []
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            if cleaned and "?" in cleaned:
                questions.append(cleaned)
        return questions

    def _template_questions(self, fact: str) -> List[str]:
        words = fact.split()
        short = " ".join(words[:8])
        return [
            f"What can you tell me about {short}?",
            f"Explain: {short}.",
            f"How does this work: {short}?",
            f"Tell me about {short}.",
            f"What do we know about {short}?",
        ]

    def generate_responses(
        self, query: str, knowledge_texts: List[str],
        value_joy: float = 0.5, value_curiosity: float = 0.1, value_pain: float = 0.0,
    ) -> str:
        """
        Generate a faithful response using ONLY provided knowledge.

        All responses are direct copies of knowledge — no paraphrasing.
        The model's job is to copy faithfully, not rearrange creatively.
        """
        if not knowledge_texts:
            return random.choice(NO_KNOWLEDGE_RESPONSES)

        # Direct copy — single or multi-fact
        return " ".join(knowledge_texts)

    def generate_cross_domain_pairs(self, facts: List[str], count: int = 200) -> List[TrainingExample]:
        """Generate questions that span 2-3 facts."""
        teacher = self._get_teacher()
        examples = []

        for _ in range(count):
            selected = random.sample([f for f in facts if not f.startswith("STRATEGY:")], min(3, len(facts)))

            prompt = (
                f"[INST] Given these facts:\n"
                + "\n".join(f"- {f}" for f in selected) +
                f"\n\nWrite ONE question that requires combining information from these facts.\n"
                f"Output ONLY the question. [/INST]\n"
            )

            try:
                question = teacher.generate(prompt, max_new_tokens=60).strip()
                if not question.endswith("?"):
                    question += "?"

                response = self.generate_responses(question, selected, value_joy=0.6, value_curiosity=0.2)
                examples.append(TrainingExample(
                    query=question, knowledge=selected,
                    value_joy=0.6, value_curiosity=0.2, value_pain=0.0,
                    confidence="medium", mode="slow", response=response,
                ))
            except Exception:
                continue

            if len(examples) % 50 == 0:
                print(f"  Cross-domain: {len(examples)}/{count}")

        return examples

    def generate_false_fact_pairs(self) -> List[TrainingExample]:
        """
        Generate training pairs with deliberately FALSE knowledge.
        The correct response faithfully uses the false fact.

        This teaches the model: "whatever is in <knowledge>, copy it faithfully."
        Critical for handling novel/unseen knowledge at inference time.
        """
        examples = []
        for question, false_fact in FALSE_FACTS:
            examples.append(TrainingExample(
                query=question, knowledge=[false_fact],
                value_joy=0.8, value_curiosity=0.1, value_pain=0.0,
                confidence="high", mode="fast",
                response=false_fact,
            ))
        return examples

    def generate_no_knowledge_pairs(self, count: int = 100) -> List[TrainingExample]:
        """Generate diverse 'I don't know' examples."""
        teacher = self._get_teacher()

        prompt = (
            f"[INST] Generate {count} diverse questions about random topics that a system with limited "
            f"knowledge would NOT be able to answer. Include questions about: recipes, sports scores, "
            f"personal opinions, future predictions, entertainment, fashion, politics, local events, "
            f"specific people, specific dates.\n"
            f"Output ONLY the questions, one per line, numbered. [/INST]\n"
        )

        try:
            raw = teacher.generate(prompt, max_new_tokens=2000)
            questions = self._parse_questions(raw)
        except Exception:
            questions = []

        # Fallback fixed questions
        fallback = [
            "What is the meaning of life?",
            "Who won the world series in 2024?",
            "What is the best programming language?",
            "How do I bake a chocolate cake?",
            "What will happen tomorrow?",
            "Tell me about quantum computing.",
            "What color is happiness?",
            "How many stars are in the sky?",
            "What should I eat for dinner?",
            "Who is the president of Mars?",
            "What is the stock price of Apple?",
            "How do I fix my car?",
            "What is the best movie ever made?",
            "Can you write me a poem?",
            "What time is it in Tokyo?",
            "How tall is the Eiffel Tower?",
            "What did Einstein dream about?",
            "Who will win the next election?",
            "What is the recipe for happiness?",
            "Tell me a joke.",
        ]
        questions = (questions + fallback)[:count]

        examples = []
        for q in questions:
            examples.append(TrainingExample(
                query=q, knowledge=[],
                value_joy=0.0, value_curiosity=0.4, value_pain=0.2,
                confidence="low", mode="slow",
                response=random.choice(NO_KNOWLEDGE_RESPONSES),
            ))
        return examples

    def generate_base_dataset(
        self, facts: List[str], output_path: Path, verbose: bool = True,
    ) -> List[TrainingExample]:
        """Generate the full base training dataset."""
        examples = []

        # 1. Standard Q&A from facts (~600 examples)
        if verbose:
            print("Step 1: Generating questions for each fact...")
        fact_questions = self.generate_questions(facts)
        total_q = sum(len(fq["questions"]) for fq in fact_questions)
        if verbose:
            print(f"  Generated {total_q} questions from {len(fact_questions)} facts.")
            print("Step 2: Generating target responses...")

        count = 0
        for fq in fact_questions:
            fact = fq["fact"]
            for question in fq["questions"]:
                response = self.generate_responses(
                    question, [fact], value_joy=0.8, value_curiosity=0.1,
                )
                examples.append(TrainingExample(
                    query=question, knowledge=[fact],
                    value_joy=0.8, value_curiosity=0.1, value_pain=0.0,
                    confidence="high", mode="fast", response=response,
                ))
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Generated {count} Q&A pairs...")

        # 2. False fact pairs (~40 examples) — teaches "copy whatever knowledge says"
        if verbose:
            print("Step 3: Generating false fact pairs...")
        false_facts = self.generate_false_fact_pairs()
        examples.extend(false_facts)
        if verbose:
            print(f"  Generated {len(false_facts)} false fact pairs.")

        # 3. Cross-domain pairs (~200 examples)
        if verbose:
            print("Step 4: Generating cross-domain pairs...")
        cross = self.generate_cross_domain_pairs(facts, count=200)
        examples.extend(cross)
        if verbose:
            print(f"  Generated {len(cross)} cross-domain pairs.")

        # 4. No-knowledge pairs (~100 examples)
        if verbose:
            print("Step 5: Generating no-knowledge pairs...")
        no_knowledge = self.generate_no_knowledge_pairs(count=100)
        examples.extend(no_knowledge)
        if verbose:
            print(f"  Generated {len(no_knowledge)} no-knowledge pairs.")

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")
        if verbose:
            print(f"  Saved {len(examples)} base examples to {output_path}")

        return examples

    def augment_dataset(
        self, base_examples: List[TrainingExample], target_size: int = 60000,
    ) -> List[TrainingExample]:
        """Augment base examples with variations."""
        augmented = list(base_examples)
        no_knowledge_examples = [ex for ex in base_examples if not ex.knowledge]
        knowledge_examples = [ex for ex in base_examples if ex.knowledge]

        while len(augmented) < target_size:
            # 15% chance of generating a no-knowledge variation
            if random.random() < 0.15 and no_knowledge_examples:
                base = random.choice(no_knowledge_examples)
                augmented.append(TrainingExample(
                    query=base.query, knowledge=[],
                    value_joy=0.0,
                    value_curiosity=random.uniform(0.1, 0.6),
                    value_pain=random.uniform(0.0, 0.4),
                    confidence="low",
                    mode=random.choice(["slow", "light"]),
                    response=base.response,
                ))
                continue

            if not knowledge_examples:
                break

            base = random.choice(knowledge_examples)
            variation = random.choice([
                "shuffle", "vary_values", "change_mode",
            ])

            if variation == "shuffle" and len(base.knowledge) > 1:
                augmented.append(TrainingExample(
                    query=base.query,
                    knowledge=random.sample(base.knowledge, len(base.knowledge)),
                    value_joy=base.value_joy, value_curiosity=base.value_curiosity,
                    value_pain=base.value_pain, confidence=base.confidence,
                    mode=base.mode, response=base.response,
                ))
            elif variation == "vary_values":
                joy = random.uniform(0.0, 1.0)
                conf = "high" if joy > 0.6 else ("medium" if joy > 0.3 else "low")
                augmented.append(TrainingExample(
                    query=base.query, knowledge=base.knowledge,
                    value_joy=joy,
                    value_curiosity=random.uniform(0.0, 0.6),
                    value_pain=random.uniform(0.0, 0.3),
                    confidence=conf, mode=base.mode, response=base.response,
                ))
            elif variation == "change_mode":
                augmented.append(TrainingExample(
                    query=base.query, knowledge=base.knowledge,
                    value_joy=base.value_joy, value_curiosity=base.value_curiosity,
                    value_pain=base.value_pain, confidence=base.confidence,
                    mode=random.choice(["fast", "light", "slow"]),
                    response=base.response,
                ))

        return augmented[:target_size]

    def prepare_training_data(self, examples: List[TrainingExample], output_path: Path):
        """Save examples as JSONL with encoder_input and decoder_target fields."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps({
                    "encoder_input": ex.to_encoder_input(),
                    "decoder_target": ex.to_decoder_target(),
                }) + "\n")
        print(f"Saved {len(examples)} training examples to {output_path}")

    def pretrain_copy_task(
        self,
        output_data_path: Path,
        output_model_path: Optional[Path] = None,
        num_samples: int = 100000,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        warmup_steps: int = 300,
        max_len: int = 128,
    ):
        """
        Phase 1: Copy pretraining on diverse English sentences.

        Downloads AG News dataset (120K news sentences across world/sports/
        business/science). Trains MiniT5 to copy sentences verbatim:

            Encoder: "The economy grew by 3.5 percent."
            Decoder: "<bos> The economy grew by 3.5 percent. <end>"

        This teaches the model English grammar, word order, and vocabulary
        through copying — zero factual knowledge is memorized.

        Returns training history dict.
        """
        from datasets import load_dataset

        output_model_path = output_model_path or self.config.decoder_model_path

        # Download AG News
        print("Downloading AG News dataset...")
        ds = load_dataset("ag_news", split="train")
        print(f"  Raw dataset: {len(ds)} examples")

        # Filter: take sentences ≤ 40 words, clean up
        examples = []
        for item in ds:
            text = item["text"].strip()
            # Take first sentence if multiple
            if ". " in text:
                text = text[:text.index(". ") + 1]
            words = text.split()
            if 5 <= len(words) <= 40:
                examples.append(text)
            if len(examples) >= num_samples:
                break

        print(f"  Filtered to {len(examples)} sentences (5-40 words)")

        # Format as JSONL: encoder_input = sentence, decoder_target = <bos> sentence <end>
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_data_path, "w") as f:
            for text in examples:
                f.write(json.dumps({
                    "encoder_input": text,
                    "decoder_target": f"{BOS_TOKEN} {text} {END_TOKEN}",
                }) + "\n")
        print(f"  Saved copy pretraining data to {output_data_path}")

        # Train
        print("\n=== Phase 1: Copy Pretraining ===")
        history = self.train(
            training_data_path=output_data_path,
            output_model_path=output_model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_input_len=max_len,
            max_output_len=max_len,
            checkpoint_name="model_phase1.pt",
        )
        return history

    def train(
        self,
        training_data_path: Path,
        output_model_path: Optional[Path] = None,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        max_input_len: int = 256,
        max_output_len: int = 64,
        val_split: float = 0.05,
        resume_from: Optional[Path] = None,
        checkpoint_name: str = "model_best.pt",
    ):
        """
        Train MiniT5 on the generated dataset.

        Args:
            resume_from: Path to a checkpoint file (.pt) to resume from.
                         Used for Phase 2 to load Phase 1 weights.
            checkpoint_name: Name for the best checkpoint file.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        output_model_path = output_model_path or self.config.decoder_model_path
        device = self.config.get_device()

        # Get tokenizer
        tokenizer = Decoder.get_tokenizer()
        pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

        # Build model
        model_config = MiniT5Config(
            vocab_size=len(tokenizer),
            d_model=self.config.decoder_d_model,
            n_heads=self.config.decoder_n_heads,
            n_encoder_layers=self.config.decoder_n_encoder_layers,
            n_decoder_layers=self.config.decoder_n_decoder_layers,
            d_ff=self.config.decoder_d_ff,
            max_seq_len=self.config.decoder_max_seq_len,
            dropout=self.config.decoder_dropout,
            pad_token_id=pad_id,
        )
        model = MiniT5(model_config).to(device)
        print(f"MiniT5 parameters: {model.param_count():,}")

        # Resume from checkpoint if provided
        if resume_from is not None:
            print(f"Loading checkpoint from {resume_from}")
            state_dict = torch.load(resume_from, map_location=device)
            model.load_state_dict(state_dict)
            print("  Checkpoint loaded.")

        # Load data
        print(f"Loading training data from {training_data_path}")
        data = []
        with open(training_data_path) as f:
            for line in f:
                data.append(json.loads(line))
        print(f"  Loaded {len(data)} examples")

        # Tokenize
        class Seq2SeqDataset(Dataset):
            def __init__(self, examples, tokenizer, max_src, max_tgt, pad_id):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_src = max_src
                self.max_tgt = max_tgt
                self.pad_id = pad_id

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                ex = self.examples[idx]
                src = self.tokenizer(
                    ex["encoder_input"], truncation=True,
                    max_length=self.max_src, padding="max_length",
                    return_tensors="pt",
                )
                tgt = self.tokenizer(
                    ex["decoder_target"], truncation=True,
                    max_length=self.max_tgt, padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "src_ids": src["input_ids"].squeeze(0),
                    "tgt_ids": tgt["input_ids"].squeeze(0),
                }

        # Split
        random.shuffle(data)
        val_size = max(1, int(len(data) * val_split))
        val_data = data[:val_size]
        train_data = data[val_size:]

        train_ds = Seq2SeqDataset(train_data, tokenizer, max_input_len, max_output_len, pad_id)
        val_ds = Seq2SeqDataset(val_data, tokenizer, max_input_len, max_output_len, pad_id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, warmup_steps + 1),
        )
        # NLL loss on pointer-generator probability distribution
        nll_criterion = torch.nn.NLLLoss(ignore_index=pad_id)

        # Training loop
        print(f"Training for {epochs} epochs, {len(train_loader)} batches/epoch")
        output_model_path.mkdir(parents=True, exist_ok=True)
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)

                src_pad_mask = (src_ids == pad_id)
                tgt_input = tgt_ids[:, :-1]
                tgt_labels = tgt_ids[:, 1:]
                tgt_pad_mask = (tgt_input == pad_id)

                # forward() returns probabilities from pointer-generator
                final_dist = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
                log_probs = torch.log(final_dist)
                loss = nll_criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_labels.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    avg = total_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)}, loss: {avg:.4f}")

            avg_train = total_loss / len(train_loader)
            history["train_loss"].append(avg_train)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    src_ids = batch["src_ids"].to(device)
                    tgt_ids = batch["tgt_ids"].to(device)
                    src_pad_mask = (src_ids == pad_id)
                    tgt_input = tgt_ids[:, :-1]
                    tgt_labels = tgt_ids[:, 1:]
                    tgt_pad_mask = (tgt_input == pad_id)
                    final_dist = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
                    log_probs = torch.log(final_dist)
                    loss = nll_criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_labels.reshape(-1))
                    val_loss += loss.item()
            avg_val = val_loss / max(len(val_loader), 1)
            history["val_loss"].append(avg_val)

            print(f"Epoch {epoch+1}/{epochs} — train: {avg_train:.4f}, val: {avg_val:.4f}")

            # Save best model by validation loss
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), output_model_path / checkpoint_name)
                print(f"  Saved best model (val_loss={avg_val:.4f})")

        # Save final model + copy best as the default
        print(f"Saving model to {output_model_path}")
        torch.save(model.state_dict(), output_model_path / "model_final.pt")
        tokenizer.save_pretrained(str(output_model_path))

        # Use best model as the default
        import shutil
        best_path = output_model_path / checkpoint_name
        if best_path.exists():
            shutil.copy(best_path, output_model_path / "model.pt")
            print(f"Using best model (val_loss={best_val_loss:.4f}) as model.pt")

        # Save training history
        with open(output_model_path / "history.json", "w") as f:
            json.dump(history, f)

        print(f"Training complete. Model size: {sum(p.numel() for p in model.parameters()):,} params")
        return history
