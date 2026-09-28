

from chunking import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "you don't have enough information -- do not make anything up. "
    "Cite which source each fact comes from using the [source] tags given."
)


def build_prompt(question: str, retrieved: list[tuple[float, Chunk]]) -> list[dict]:
    """Builds a chat-formatted prompt (list of role/content dicts)."""
    context_blocks = []
    for score, chunk in retrieved:
        context_blocks.append(
            f"[source: {chunk.source}#{chunk.chunk_id}]\n{chunk.text}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class LocalGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 400,
        device: str | None = None,
    ):
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else "auto",
        ).to(self.device)

    def generate(self, question: str, retrieved: list[tuple[float, Chunk]]) -> str:
        import torch

        messages = build_prompt(question, retrieved)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,        # deterministic; set True + temperature for variety
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
