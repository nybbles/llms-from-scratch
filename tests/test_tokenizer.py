import tiktoken


def test_bpe(sample_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(sample_text)
    strings = tokenizer.decode(integers)

    print(f"Encoded {len(integers)} tokens")

    context_size = 4
