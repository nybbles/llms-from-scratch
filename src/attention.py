import torch


def dot_product_attention_weights(inputs):
    attention_scores = inputs @ inputs.transpose(-1, -2)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return attention_weights


def apply_attention_weights(attention_weights, inputs):
    return attention_weights @ inputs


def scaled_dot_product_attention(
    inputs,
    query_embedder: torch.nn.Module,
    key_embedder: torch.nn.Module,
    value_embedder: torch.nn.Module,
):
    queries = query_embedder(inputs)
    keys = key_embedder(inputs)
    values = value_embedder(inputs)
    attention_scores = queries @ keys.transpose(-1, -2)
    attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

    return attention_weights @ values
