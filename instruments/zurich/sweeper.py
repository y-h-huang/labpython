from .utils import HFModule, add_class_prop
import numpy as np

class HFSweeper(HFModule):
    
    def __init__(self, *args):
        super().__init__(*args)
        self.subscribe('demods/0/sample')

    def sweep(self, start, stop, npoints, linear=True):
        self.set({
                'start': start,
                'stop': stop,
                'samplecount': npoints,
                'xmapping': 0 if linear else 1,
            })

        return self.read()[0][0][0]

    def __del__(self):
        self.unsubscribe('*')

