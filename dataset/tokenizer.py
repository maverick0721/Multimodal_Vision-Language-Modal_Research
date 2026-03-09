import tiktoken

class Tokenizer:

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self,text):
        return self.enc.encode(text)

    def decode(self,tokens):
        return self.enc.decode(tokens)