import tiktoken
import torch

from data import create_dataloader_v1
from embedding import create_embedder_v1

torch.manual_seed(42)


def test_embedding_model(sample_text):
    batch_size = 4
    max_length = 256
    embedding_dim = 256

    tokenizer = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader_v1(
        sample_text,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    embedder = create_embedder_v1(
        tokenizer, embedding_dim=embedding_dim, context_length=max_length
    )

    first_batch = next(iter(dataloader))
    input_ids, target_ids = first_batch

    embedded = embedder(input_ids)
    assert embedded.shape == (batch_size, max_length, embedding_dim)
