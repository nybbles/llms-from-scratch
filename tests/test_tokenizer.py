import tiktoken
from pytest import fixture


@fixture
def sample_text():
    with open("tests/The_Verdict.txt", "r") as file:
        text = file.read()
        yield text


def test_bpe(sample_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(sample_text)
    strings = tokenizer.decode(integers)

    print(f"Encoded {len(integers)} tokens")

    context_size = 4
