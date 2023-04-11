import zhinst.core as zi
import time
import numpy as np
from .utils import settings_tree, add_class_prop


class HF2LI:

    def __init__(self, device_id, settings={}, host='localhost', port=8005):
        self.device_id = device_id
        self.daq = zi.ziDAQServer(host, port, 1) # 1: API level

        self.set(settings)

        self.prop_array('sigins', 2, 'range ac imp50 diff')
        self.prop_array('sigouts', 2, 'on add range offset')
        self.prop_array('oscs', 6, 'freq')
        self.prop_array('demods', 6, 'adcselect order timeconstant rate trigger enable oscselect harmonic freq phaseshift sinc sample')
        self.prop_array('plls', 2, 'adcselect order timeconstant rate trigger enable oscselect harmonic freq phaseshift sinc sample')
        self.prop_array('scopes', 1, 'wave channel time bwlimit trigchannel enable trigedge triglevel trigholdoff')
        self.prop_array('dios', 1, 'extclk decimation drive output syncselect0 syncselect1 input')


    def prop_array(self, prefix, count, suffixes):
        if isinstance(suffixes, str):
            suffixes = suffixes.split()

        res = []
        for n in range(count):
            class X:
                def __init__(self, daq):
                    self.daq = daq

            for s in suffixes:
                add_class_prop(X, s, f'{prefix}/{n}/{s}', via_daq=True)

            res.append(X(self))

        setattr(self, prefix, res)


    def path(self, p):
        return f'/{self.device_id}/{p}'


    def set(self, d, v=None):
        if v is not None:
            d = {d: v}

        self.daq.set(list(settings_tree({self.device_id: d})))
        self.daq.sync()

        return self


    def get(self, key):
        k = self.path(key)
        res = self.daq.get(k, True)[k]

        if isinstance(res, np.ndarray) and len(res) == 1:
            res = res[0]

        return res


    def scope(self):
        from .scope import HFScope
        return HFScope(self, self.daq.scopeModule())


    def spectrum(self):
        from .spectrum import HFSpectrum
        spec = HFSpectrum(self, self.daq.dataAcquisitionModule())

        return spec

    def sweeper(self):
        from .sweeper import HFSweeper
        sw = HFSweeper(self, self.daq.sweep())

        return sw


add_class_prop(HF2LI, 'status', 'status/flags/binary')
add_class_prop(HF2LI, 'clockbase', 'clockbase')
