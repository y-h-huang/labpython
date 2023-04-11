import numpy as np

def settings_tree(s, prefix='/'):
    for k, v in s.items():
        k = prefix + k
        if not isinstance(v, dict):
            yield k, v
        else:
            yield from settings_tree(v, k + '/')

def add_class_prop(cls, propname, path, via_daq=False):
    def mod_prop(s):
        @property
        def p(self):
            return self.get(s)

        @p.setter
        def p(self, v):
            self.set(s, v)

        return p

    def daq_prop(s):
        @property
        def p(self):
            return self.daq.get(s)

        @p.setter
        def p(self, v):
            self.daq.set(s, v)

        return p

    setattr(cls, propname, daq_prop(path) if via_daq else mod_prop(path))


class HFModule:

    def __init__(self, daq, mod):
        self.daq = daq
        self.mod = mod
        self.subs = []
        self.set('device', daq.device_id)


    @classmethod
    def mod_prop(cls, **kwargs):
        for k, v in kwargs.items():
            add_class_prop(cls, k, v, via_daq=False)


    @classmethod
    def daq_prop(cls, **kwargs):
        for k, v in kwargs.items():
            add_class_prop(cls, k, v, via_daq=True)


    def set(self, d, v=None):
        if v is not None:
            d = {d: v}

        for k, v in settings_tree(d, prefix=''):
            self.mod.set(k, v)

        return self

    def get(self, k):
        res = self.mod.get(k, True)['/' + k]
        if isinstance(res, np.ndarray) and len(res) == 1:
            res = res[0]
        return res


    def read(self,  error_free=False):

        while True:
            if error_free:
                self.daq.status = 0
                s = 0

            self.mod.execute()
            while (prog := float(self.mod.progress())) != 1:
                if error_free:
                    s |= self.daq.status

            res = self.mod.read(True)

            if not error_free or not s:
                return [res[k] for k in self.subs]


    def subscribe(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        keys = [self.daq.path(k) for k in keys]

        for k in keys:
            if k not in self.subs:
                self.subs.append(k)
                self.mod.subscribe(k)

        return self

    def unsubscribe(self, keys):
        if isinstance(keys, str):
            if keys == '*':
                self.mod.unsubscribe('*')
                self.subs = []
                return
            else:
                keys = [keys]

        self.subs = [k for k in self.subs if k not in keys]
        for k in keys:
            self.mod.unsubscribe(k)

        return self

    def __del__(self):
        self.mod.unsubscribe('*')

