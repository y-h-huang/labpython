from .utils import HFModule

class HFScope(HFModule):
    def __init__(self, daq, mod):
        super().__init__(daq, mod)

        clockbase = self.daq.clockbase
        self.rates = [clockbase >> n for n in range(16)]

        self.subscribe('scopes/0/wave')


    def sampling_rate(self, rate=None):
        key = 'scopes/0/time'

        if rate == None:
            return self.rates[self.daq.scopes[0].time]

        for i, v in enumerate(self.rates):
            if v < rate:
                i -= 1
                break

        self.daq.scopes[0].time = max(i, 0)


    def read(self, channel, fft=False, error_redo=False):
        self.mode = 3 if fft else 1

        if channel < 2:
            self.externalscaling = self.daq.sigins[channel].range
        else:
            self.externalscaling = self.daq.sigouts[channel - 2].range

        self.daq.scopes[0].channel = channel

        self.enable = 1
        self.mod.execute()

        while True:
            if error_redo:
                self.daq.status, s = 0, 0

            while (prog := float(self.mod.progress())) != 1 and self.records < 1:
                #rec = self.mod.getInt('records')
                #print(f'progress={prog} records={self.records}')
                if error_redo:
                    s |= self.daq.status

            res = self.mod.read(True)

            if not error_redo or not s:
                res = [res[k] for k in self.subs][0][0][0]['wave'][0]
                break

        self.mod.finish()
        self.enable = 0
        return res

HFScope.daq_prop(
        bwlimit='scopes/0/bwlimit',
        enable='scopes/0/enable',
    )
HFScope.mod_prop(
        mode='mode',
        externalcaling='externalscaling',
        records='records',
    )
