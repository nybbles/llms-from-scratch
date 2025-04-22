import torch

from attention import dot_product_attention_weights


def test_dot_product_attention(batch_size, embedding_dim, first_batch, embedder):
    input_ids, target_ids = first_batch
    embedded = embedder(input_ids)
    attention_weights = dot_product_attention_weights(embedded)
    assert attention_weights.shape == torch.Size([batch_size, embedding_dim, embedding_dim])
