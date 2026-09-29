
from PIL import Image


class ImageUnderstander:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        # Lazy-imported so the rest of the pipeline works without these
        # (larger) dependencies unless image support is actually used.
        import torch
        from transformers import AutoProcessor

        # The class name for this changed between transformers versions:
        # older releases call it AutoModelForVision2Seq, newer ones renamed
        # it to AutoModelForImageTextToText. Try the current name first,
        # fall back to the old one so this works either way.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = _AutoVLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)

    def _ask(self, image_path: str, question: str, max_new_tokens: int = 500) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)

        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Only decode the newly generated tokens, not the echoed prompt.
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def describe(self, image_path: str) -> str:
        """A general, detailed description -- used to index the image
        for retrieval (this is what gets embedded and searched)."""
        return self._ask(
            image_path,
            "Describe this image in detail, including any text, numbers, "
            "labels, charts, diagrams, or data shown.",
        )

    def answer(self, image_path: str, question: str) -> str:
        """Answer a SPECIFIC question directly about the image -- used
        at query time when this image is the top retrieved match."""
        return self._ask(image_path, question)