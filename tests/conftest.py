import tiktoken
from pytest import fixture

from data import create_dataloader_v1
from embedding import create_embedder_v1


@fixture(scope="module", params=[4])
def batch_size(request):
    return request.param


@fixture(scope="module", params=[128])
def max_length(request):
    return request.param


@fixture(scope="module", params=[256])
def embedding_dim(request):
    return request.param


@fixture(scope="module", params=["gpt2"])
def encoding_type(request):
    return request.param


@fixture
def sample_text():
    with open("tests/The_Verdict.txt", "r") as file:
        text = file.read()
        yield text


@fixture
def tokenizer(encoding_type):
    return tiktoken.get_encoding(encoding_type)


@fixture
def dataloader(sample_text, batch_size, max_length, embedding_dim, tokenizer):
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

    return dataloader


@fixture
def embedder(tokenizer, max_length, embedding_dim):
    return create_embedder_v1(tokenizer, embedding_dim=embedding_dim, context_length=max_length)


@fixture
def first_batch(dataloader):
    return next(iter(dataloader))
