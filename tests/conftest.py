from pytest import fixture


@fixture
def sample_text():
    with open("tests/The_Verdict.txt", "r") as file:
        text = file.read()
        yield text
