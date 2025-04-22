import tiktoken
import torch

from data import create_dataloader_v1
from embedding import create_embedder_v1

torch.manual_seed(42)


def test_embedding_model(sample_text, tokenizer, dataloader, batch_size, max_length, embedding_dim):
    embedder = create_embedder_v1(tokenizer, embedding_dim=embedding_dim, context_length=max_length)

    first_batch = next(iter(dataloader))
    input_ids, target_ids = first_batch

    embedded = embedder(input_ids)
    assert embedded.shape == (batch_size, max_length, embedding_dim)
