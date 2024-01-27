import numpy as np
import pyvisa
from utils import load_data
import matplotlib.pyplot as plt
from instruments.srs_dc205 import DC205
from ultraview_daq import get_data
import time, sys

def get_cal_data(chan):
    dc = DC205(address='COM4')
    dc.floating(True)
    dc.output(False)

    fig, ax = plt.subplots(2, 1)

    plt.show(block=False)

    V = []
    avg = []
    std = []
    for v in np.linspace(-0.25, 0.25, 31):
        dc.voltage(v)
        dc.output(True)
        time.sleep(1)

        d = get_data(n_blocks=10, channels=[chan], in_volts=False)
        V.append(v)

        adc = np.asfarray(d[0])
        avg.append(np.mean(adc))
        std.append(np.std(adc))

        ax[1].clear()
        ax[1].plot(adc)

        print(V[-1], avg[-1])

        ax[0].clear()
        ax[0].errorbar(V, avg, yerr=std, fmt='.')

        fig.canvas.draw()
        fig.canvas.flush_events()
    np.save(f'calib{chan}.npy', {
                'V': np.array(V), 'avg': np.array(avg), 'std': np.array(std)
            })
    dc.output(False)


def calibrate():
    for i in range(4):
        d = load_data(f'calib{i}.npy', 'Data')
        p = np.polyfit(d.avg, d.V, deg=1)
        x = np.array([0, 2**16 - 1])
        y = np.polyval(p, x)

        print(i, p)

        plt.plot(d.avg, d.V, '.')
        plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    cmd = sys.argv[1]

    match cmd:
        case 'get':
            chan = int(sys.argv[2])
            get_cal_data(chan)

        case 'fit':
            calibrate()

        case others:
            raise ValueError(f'Unknown command {cmd}')


