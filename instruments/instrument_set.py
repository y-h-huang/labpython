from contextlib import contextmanager
from collections import namedtuple
from tkinter import W
import pyvisa

class InstrumentSet():
    def __init__(self, *lst):
        self.devs = []
        self.shutdown_hooks = []

        for l in map(list, lst):
            if len(l) < 4:
                l.append(False)
            self.devs.append(l)

    def add(self, name, cls, addr, inactive=False):
        self.devs.append((name, cls, addr, inactive))
        return self

    def at_shutdown(self, func):
        self.shutdown_hooks.append(func)
        return self

    @staticmethod
    def run_with_idns(callback, **tags):
        # tags is a name->(cls, *strings) dictionary

        from instruments.devicelist import device_ids
        iset = InstrumentSet()

        for addr, idn in device_ids():
            if idn is None:
                continue

            for k, v in tags.items():
                vv = [v[1]] if isinstance(v[1], str) else v[1]

                if all(idn.find(s) != -1 for s in vv):
                    del tags[k] # remove from candidates
                    iset.add(name=k, cls=v[0], addr=addr)
                    break

        with iset.run() as d:
            callback(d)

    @contextmanager
    def run(self):
        devs = []

        rm = None
        dev_set = namedtuple('InstrumentSet', [x[0] for x in self.devs])

        try:
            if any(not x[-1] for x in self.devs):
                rm = pyvisa.ResourceManager()

            for name, cls, addr, inactive in self.devs:
                if inactive:
                    devs.append(None)
                else:
                    devs.append(cls(rm, addr))
            yield dev_set(*devs)

        finally:
            for f in self.shutdown_hooks:
                f()

            for d in devs:
                if d is not None:
                    d.close()

            if rm: rm.close()
