import torch

from attention import apply_attention_weights, dot_product_attention_weights


def test_dot_product_attention(batch_size, max_length, first_batch, embedder):
    input_ids, target_ids = first_batch
    embedded = embedder(input_ids)
    attention_weights = dot_product_attention_weights(embedded)

    # Attention weights should be max_length x max_length because it captures
    # self-attention of attending to all tokens at every token position.
    assert attention_weights.shape == torch.Size([batch_size, max_length, max_length])
    assert torch.all(attention_weights.sum(dim=-1) == 1)


def test_apply_attention_weights(batch_size, max_length, embedding_dim, first_batch, embedder):
    input_ids, target_ids = first_batch
    embedded = embedder(input_ids)
    attention_weights = dot_product_attention_weights(embedded)

    # Context vectors should be max_length x embedding_dim because each context
    # vector has the same dimension as the input and there is one context
    # vector for each position in the input sequence.
    context_vectors = apply_attention_weights(attention_weights, embedded)
    assert context_vectors.shape == torch.Size([batch_size, max_length, embedding_dim])
