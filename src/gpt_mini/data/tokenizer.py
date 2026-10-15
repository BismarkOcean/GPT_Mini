class Tokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))

        self.char_to_id = {}
        self.id_to_char = {}

        for i, ch in enumerate(chars):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch

    # 编码函数
    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self.char_to_id:
                raise ValueError(f"未知字符:{ch}")
            ids.append(self.char_to_id[ch])
        return ids

    # 解码函数
    def decode(self, ids: list[int]) -> str:
        text = ""

        for i in ids:
            if type(i) is not int:
                raise TypeError(f"ID类型错误: {type(i).__name__}")
            if i not in self.id_to_char:
                raise ValueError(f"非法 ID: {i}")

            text += self.id_to_char[i]

        return text
