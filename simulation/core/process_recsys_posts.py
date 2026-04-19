"""
process_recsys_posts.py — batch embedding helper for TWHIN-BERT recsys.

GPU improvements applied:
- fp16 half-precision inference (model.half()) when CUDA is available
- non_blocking=True for async CPU→GPU transfers
- Dynamic batch size: 128 on GPU, 32 on CPU
"""
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_BATCH_SIZE = 128 if _device.type == "cuda" else 32


def _apply_gpu_optimizations(model: AutoModel) -> AutoModel:
    """Cast to fp16 when running on CUDA for faster inference and lower VRAM usage."""
    if _device.type != "cuda":
        return model
    model = model.half()
    return model


@torch.no_grad()
def process_batch(model: AutoModel, tokenizer: AutoTokenizer,
                  batch_texts: List[str]):
    inputs = tokenizer(batch_texts,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    inputs = {k: v.to(_device, non_blocking=True) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.pooler_output


def generate_post_vector(model: AutoModel, tokenizer: AutoTokenizer, texts,
                         batch_size=None):
    if batch_size is None:
        batch_size = _DEFAULT_BATCH_SIZE
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_outputs = process_batch(model, tokenizer, batch_texts)
        all_outputs.append(batch_outputs)
    return torch.cat(all_outputs, dim=0).cpu()


def generate_post_vector_openai(texts: List[str], batch_size: int = 100):
    """Generate embeddings using OpenAI API (requires camel-ai)."""
    from camel.embeddings import OpenAIEmbedding
    from camel.types import EmbeddingModelType
    openai_embedding = OpenAIEmbedding(
        model_type=EmbeddingModelType.TEXT_EMBEDDING_3_SMALL)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned_texts = [
            text.strip() if text and isinstance(text, str) else "empty"
            for text in batch_texts
        ]
        batch_embeddings = openai_embedding.embed_list(objs=cleaned_texts)
        all_embeddings.append(torch.tensor(batch_embeddings))
    return torch.cat(all_embeddings, dim=0)
