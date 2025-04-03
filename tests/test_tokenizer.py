import tiktoken

def test_bpe():
    tokenizer = tiktoken.get_encoding("gpt2")
    test = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
        "of someunknownPlace"
    )
    integers = tokenizer.encode(test, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)
