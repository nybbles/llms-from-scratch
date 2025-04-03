import torch


def create_embedder_v1(tokenizer, embedding_dim, context_length=0):
    vocab_size = tokenizer.n_vocab
    embedder = torch.nn.Embedding(vocab_size, embedding_dim)

    if context_length == 0:
        return embedder

    positional_embedder = torch.nn.Embedding(context_length, embedding_dim)

    def positional_embedding(input_ids):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
        return embedder(input_ids) + positional_embedder(positions)

    return positional_embedding
