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
