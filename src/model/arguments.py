import json


class Arguments(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return self.print()

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return getattr(self, key, None)

    def print(self, offs=0):
        st = ''
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                st += '\n<br>{} {} = {}'.format('-' * offs, k, v.print(offs=offs + 3))
            elif isinstance(v, type):
                st += '\n<br>{} {} = {}'.format('-' * offs, k, str(v.__class__))
            else:
                st += '\n<br>{} {} = {}'.format('-' * offs, k, str(v))
        return st

    def __str__(self):
        return self.print()

    def __len__(self):
        return len(self.__dict__)

    def save(self):
        js = {}


    def load(self):
        pass

