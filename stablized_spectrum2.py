import numpy as np
import time

from pid import PID
from instcmd.BNC845 import cmd_BNC
from instruments.rfsource import BNC845
from instruments.Keithley2400 import KEITHLEY
from instruments.vna import VNA
from instruments.instrument_set import InstrumentSet
from instruments.devicelist import device_ids
from instruments.zurich import HF2LI

from utils import AvgStdAccumulator, AvgStdAccumulatorC, fft
import matplotlib.pyplot as plt

zurich_setting = {
        'sigins': {
            '0': {
                'ac': 0,
                'imp50': 0,
                'diff': 0,
                'range': 0.1
            },
            '1': {
                'ac': 0,
                'imp50': 0,
                'diff': 0,
                'range': 0.1
            }
        },
        'demods': {
            '0': {
                'oscselect': 0,
                'adcselect': 0,
                'order': 4,
                'timeconstant': 9e-6,
                'rate': 4000,
            },
            '1': {
                'oscselect': 0,
                'adcselect': 1,
                'order': 4,
                'timeconstant': 9e-6,
                'rate': 4000,
            },
        },
    }

spectrum_setting = {
        'grid': {
            'cols': 1024,
            'mode': 4,
        },
        'delay': 0.1,
        'type': 0,
        'fft/window': 0,
    }

        #'spectrum/frequencyspan': 500,
class Experiment:

    def __init__(self, device_set, zurich):
        self.dev = device_set
        self.lockin = zurich

        # shift from bottom of hybrid peak
        self.f_offset = 0

        self.f_center = 7.102834033e9 - 100e3
        self.f_mech =  12.756400e6
        self.f_demod = 12.755500e6
        self.rf_dbm = 10
        self.block_length = 512
        self.block_count = 16

        self.measure_vna = True
        self.measure_lockin = False
        self.pid_on = True

        self.phase_target = 0
        self.initialize()

    def initialize(self):
        self.rf_freq = self.f_center + self.f_mech + self.f_offset

        for x in (self.dev.rf, self.dev.lo):
            x.auto_off = False
            x.frequency(self.rf_freq)
            x.output(True)

        self.dev.vna.output(True)

        self.fig = fig = plt.figure(figsize=(12, 5))
        fig.clear()

        self.ax_pid = fig.add_subplot(2, 3, 1)
        self.ax_peak = fig.add_subplot(2, 3, 4)
        self.ax_vna = fig.add_subplot(2, 3, (2, 3))
        self.ax_lockin = fig.add_subplot(2, 3, (5, 6), sharex=self.ax_vna)

        plt.show(block=False)

        self.create_pid()

    def vnasweep(self, f, bw, delay=240e-9):
        z = self.dev.vna.sweep(f[0], f[-1], len(f), bw, only_z=True)
        return z*np.exp(1j*delay*(f - f[0]))


    def peak_offset(self):
        from utils import chebpoly

        f_wide = np.linspace(self.f_center - 4e6, self.f_center + 4e6, 1001)

        z = self.vnasweep(f_wide, bw=1e4)

        r = np.abs(z)
        idx = np.argmin(r)

        f0 = f_wide[idx]
        sel = (f_wide > f0 - 1e6) & (f_wide < f0 + 2e6)
        f = f_wide[sel]

        rfit = chebpoly(f - f[0], r[sel], deg=3)
        f0 = f[np.argmin(rfit)]

        fig, ax = self.fig, self.ax_peak
        ax.clear()
        ax.plot(f_wide, r, lw=.5)
        ax.plot(f, rfit, lw=.5)
        ax.axvline(f0)

        fig.canvas.draw()
        fig.canvas.flush_events()

        return (f0 - self.f_center)*1e-6

    def create_pid(self):

        def measure_peak(inc):
            psu = self.dev.psu

            bias = psu.current() + inc
            psu.current(bias)
            return self.peak_offset()

        self.pid = PID('PEAK', measure_peak, Kp=5e-2, Ti=5, Td=1, threshold=5e-2, hist_len=100, fig=self.fig, ax=self.ax_pid)
        self.pid.I_decay = .6

    def lock_phase(self, threshold=1):
        self.lockin.set({'sigins/0': {'range': 0.1, 'ac': 0}, 'sigins/1': {'range': .1, 'ac': 0}})
        target = self.phase_target
        scope, lo = self.scope, self.dev.lo

        while True:
            delta = np.arctan2(np.mean(scope.read(1)), np.mean(scope.read(0)))*180/np.pi - target
            if np.abs(delta) < threshold:
                break

            lo.phase(lo.phase() + delta)

        self.lockin.set({'sigins/0': {'range': 0.02, 'ac': 1}, 'sigins/1': {'range': .02, 'ac': 1}})

    def save_data(self, avg_vna, avg_i, avg_q, f_vna, f_spec):
        np.save('save.npy', {
                    'f_vna': f_vna,
                    'f_spec': f_spec,
                    'avg_vna': avg_vna.serialize(),
                    'avg_i': avg_i.serialize(),
                    'avg_q': avg_q.serialize()
                })

    def run(self):
        lockin = self.lockin
        lockin.set(zurich_setting)
        lockin.oscs[0].freq = self.f_demod

        self.scope = scope = self.lockin.scope()
        scope.sampling_rate(10e3)

        '''
        self.spectrum = lockin.spectrum()                                   \
                              .set(spectrum_setting)                        \
                              .subscribe(['demods/0/sample.xiy.fft.real.avg',
                                          'demods/0/sample.xiy.fft.imag.avg',
                                          'demods/1/sample.xiy.fft.real.avg',
                                          'demods/1/sample.xiy.fft.imag.avg'])
        '''
        self.spectrum = lockin.spectrum()                                   \
                              .set(spectrum_setting)                        \
                              .set('grid/cols', self.block_count*self.block_length) \
                              .subscribe(['demods/0/sample.x.avg',
                                          'demods/0/sample.y.avg',
                                          'demods/1/sample.x.avg',
                                          'demods/1/sample.y.avg'])

        f_vna = np.linspace(self.f_center - 1000, self.f_center + 1000, 1001) + self.f_offset

        avg_vna = AvgStdAccumulatorC()
        avg_i = AvgStdAccumulator()
        avg_q = AvgStdAccumulator()

        last_save = None

        f_spec = self.lockin.oscs[0].freq + fft(1/self.lockin.demods[0].rate, np.zeros(self.block_length),
                                                with_f=True, with_fft=False)


        #park_freq = int(self.dev.rf.frequency() - (f_spec[0] + self.vna_park_offset))
        #park_freq = np.array([park_freq - 1, park_freq])

        from scipy.signal.windows import hann
        #window = hann(len(zi))
        window = 1

        def get_spectrum():

            l = self.block_length

            while True:
                xi, yi, xq, yq = [x[0]['value'][0] for x in self.spectrum.read(error_free=True)]

                if np.isnan(xi).any() or np.isnan(yi).any() or np.isnan(xq).any() or np.isnan(yq).any():
                    print('NaN found')
                    continue

                break

            zi = xi + 1j*yi
            zq = xq + 1j*yq

            for start in range(0, len(xi), l):

                ffti = fft(None, zi[start:start + l]*window, with_f=False)
                fftq = fft(None, zq[start:start + l]*window, with_f=False)

                avg_i.append(np.abs(ffti)**2)
                avg_q.append(np.abs(fftq)**2)

        while True:
            try:

                if self.pid_on or self.measure_vna:
                    self.dev.vna.output(True)
                    time.sleep(0.1)

                    if self.pid_on and not self.pid.adjust(): continue

                    if self.measure_vna:
                        avg_vna.append(self.vnasweep(f_vna, bw=1000))
                        self.ax_vna.clear()
                        self.ax_vna.plot(self.dev.rf.frequency() - f_vna, np.abs(avg_vna.mean()), lw=.5)

                self.dev.vna.output(False)

                # park the vna probe tone
                #self.vnasweep(park_freq, bw=7e4)

                self.lock_phase()
                time.sleep(.1)

                fig, ax = self.fig, self.ax_lockin

                if self.measure_lockin:
                    get_spectrum()

                    ax.clear()
                    ax.plot(f_spec, np.abs(avg_i.mean()), lw=0.5)
                    ax.plot(f_spec, np.abs(avg_q.mean()), lw=0.5)
                    ax.set_yscale('log')

                if self.measure_lockin or self.measure_vna:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    t = time.time()
                    if last_save is None or t > last_save + 60:
                        self.save_data(avg_vna, avg_i, avg_q, f_vna, f_spec)
                        last_save = t

            except KeyboardInterrupt:
                while True:
                    cmd, *args = input('> ').strip().split()
                    cmd = cmd.lower()

                    try:
                        if 'cont'.startswith(cmd):
                            break

                        elif 'reset'.startswith(cmd):
                            avg_i.clear()
                            avg_q.clear()
                            avg_vna.clear()

                        elif 'save'.startswith(cmd):
                            self.save_data(avg_vna, avg_i, avg_q, f_vna, f_spec)

                        elif 'pid'.startswith(cmd):
                            if args:
                                self.pid_on = bool(int(args[0]))
                            print(self.pid_on)

                        elif 'vna'.startswith(cmd):
                            if args:
                                self.measure_vna = bool(int(args[0]))
                            print(self.measure_vna)

                        elif 'lockin'.startswith(cmd):
                            if args:
                                self.measure_lockin = bool(int(args[0]))
                            print(self.measure_lockin)

                        elif 'phase'.startswith(cmd):
                            self.phase_target = float(args[0])

                        elif 'exit'.startswith(cmd):
                            return

                        else:
                            print('?')
                    except:
                        print('Garbled input')
                        pass

if __name__ == '__main__':

    iset = InstrumentSet()
    for addr, idn in device_ids():
        if idn is None:
            continue

        #print(f'Device {idn} at address {addr}')
        if idn.find('Keysight') != -1 and idn.find('E5063A') != -1:
            iset.add(name='vna', cls=VNA, addr=addr)

        elif idn.find('Berkeley Nucleonics') != -1 and idn.find('865-20') != -1:
            iset.add(name='rf', cls=BNC845, addr=addr)

        elif idn.find('Berkeley Nucleonics') != -1:
            iset.add(name='lo', cls=BNC845, addr=addr)

        elif idn.find('KEITHLEY') != -1 and idn.find('MODEL 2400') != -1:
            iset.add(name='psu', cls=KEITHLEY, addr=addr)

    with iset.run() as devs:
        exp = Experiment(devs, HF2LI(device_id='dev538'))
        exp.run()
