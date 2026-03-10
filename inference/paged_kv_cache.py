class PagedKVCache:

    def __init__(self,page_size=128):

        self.page_size = page_size
        self.pages = []

    def append(self,k,v):

        self.pages.append((k,v))

        if len(self.pages) > self.page_size:
            self.pages.pop(0)

    def get(self):

        return self.pages