import pytest

from gpt_mini.data.tokenizer import Tokenizer


def test_roundtrip():
    text = "你好 GPT!\n你好"
    tokenizer = Tokenizer(text)

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
    assert len(tokenizer.char_to_id) == len(set(text))


def test_empty():
    text = "你好 GPT!\n你好 GPT!"
    t = Tokenizer(text)
    assert t.encode("") == []
    assert t.decode([]) == ""


def test_repeat():
    text = "你好 GPT!\n你好 GPT!"
    t = Tokenizer(text)
    x = t.encode("你你")
    assert x[1] == x[0]


def test_invalid_id():
    text = "你好 GPT!\n你好 GPT!"
    t = Tokenizer(text)
    # 验证 decode([999]) 抛出 ValueError
    with pytest.raises(ValueError):
        t.decode([999])
    with pytest.raises(ValueError):
        t.decode([-1])
    # 验证 decode([-1]) 抛出 ValueError


def test_unknown_character():
    text = "你好 GPT!\n你好 GPT!"
    t = Tokenizer(text)
    with pytest.raises(ValueError):
        t.encode("啊")


def test_vocab_stability():
    text1 = Tokenizer("abcabc")
    text2 = Tokenizer("cbacba")
    assert text1.char_to_id == text2.char_to_id
    assert text2.id_to_char == text1.id_to_char


def test_invalid_id_type():
    text = "你好 GPT!\n你好 GPT!"
    t = Tokenizer(text)
    with pytest.raises(TypeError):
        t.decode([1.0])
    with pytest.raises(TypeError):
        t.decode([True])
    with pytest.raises(TypeError):
        t.decode("1")


def test_newline_keep():
    t=Tokenizer("fuck\nworld")
    assert "fuck\nworld"==t.decode(t.encode("fuck\nworld"))


