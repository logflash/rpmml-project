class NoopRenderer:
    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass

    def composite(self, *args, **kwargs):
        pass