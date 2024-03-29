from .utils import HFModule, add_class_prop
import numpy as np

class HFSpectrum(HFModule):

    def __init__(self, daq, mod):
        super().__init__(daq, mod)

        mod.set('device', daq.device_id)

        clockbase = self.daq.get('clockbase')
        self.rates = [clockbase >> n for n in range(16)]


    def time_data(self, demod_id):
        n = self.grid_cols
        dt = 1/self.daq.demods[demod_id].rate
        return dt*np.arange(n)
        
    def frequency_data(self, demod_id, absolute=True, half_span=False):

        npts = self.grid_cols
        n = npts//2

        fmax = self.daq.demods[demod_id].rate
        df = fmax/npts

        f = df*np.arange(-n + 1, n)
        if half_span:
            f = f[n - 1:]

        if not absolute:
            return f

        return self.daq.demods[demod_id].freq + f


    def read(self, subscriptions=None, unravel=True, **kwargs):
        self.enable = 1

        if subscriptions is not None:
            self.unsubscribe('*')
            self.subscribe(subscriptions)

        ret = super().read()
        if unravel:
            ret = [x[0]['value'].ravel() for x in ret]

        self.mod.finish()
        self.enable = 0

        if subscriptions is not None:
            self.unsubscribe('*')

        return ret


HFSpectrum.mod_prop(
        enable='spectrum/enable',
        grid_cols='grid/cols',
        grid_rows='grid/rows',
        grid_mode='grid/mode',
        frequencyspan='spectrum/frequencyspan',
        overlapped='spectrum/overlapped',
        endless='endless',
        autobandwidth='spectrum/autobandwidth',
        type='type',
    )
