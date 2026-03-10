class KVCache:

    def __init__(self):

        self.keys = []
        self.values = []

    def append(self,k,v):

        self.keys.append(k)
        self.values.append(v)

    def get(self):

        return self.keys,self.values