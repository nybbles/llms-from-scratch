import tiktoken


def test_bpe(sample_text, encoding_type):
    tokenizer = tiktoken.get_encoding(encoding_type)
    integers = tokenizer.encode(sample_text)
    strings = tokenizer.decode(integers)

    print(f"Encoded {len(integers)} tokens")

    context_size = 4
