import numpy as np
import subprocess, sys, os, time
from instruments.srs_dc205 import DC205
import pyvisa
import matplotlib.pyplot as plt

ACQ_PROGRAM = 'acquire.exe'
CALIBRATION = [ [ 9.29988600e-06, -3.04939374e-01],
                [ 9.29933927e-06, -3.04883729e-01],
                [ 9.18059653e-06, -3.00645034e-01],
                [ 9.18871042e-06, -3.00999036e-01] ]


def get_data(n_blocks=16,
             channels=None,
             in_volts=True,
             internal_clock=True):

    basedir = os.path.split(__file__)[0]

    cmd = [os.path.join(basedir, ACQ_PROGRAM)]

    n_blocks = int(n_blocks)

    if n_blocks > 8192:
        raise ValueError(f'Too many blocks: {n_blocks}')
    else:
        nb = 1
        while nb < n_blocks:
            nb *= 2

        cmd.append(f'{nb}')

    if internal_clock:
        cmd.append('-ic')


    if channels is None:
        channels = (0, 1, 2, 3)
    else:
        channels = tuple(int(x) for x in sorted(channels))

        match len(channels):
            case 1:
                # Channel numbers are swapped around in single channel mode somehow
                channels = [int('0231'[int(channels[0])])]
                s = str(channels[0])
                cmd.extend(['-scm', '-scs', s])
            case 2:
                s = ''.join(str(x) for x in sorted(channels))
                cmd.extend(['-dcm', '-dcs', s])

            case others:
                raise ValueError(f'Invalid channel spec {channels}')

    filename = os.path.join(basedir, 'tmp.dat')

    cmd.extend(['-f', filename])
    print('Command to be called:', ' '.join(cmd))

    subprocess.call(cmd, cwd=basedir)

    #res = np.fromfile(filename, dtype='uint16')/2**16 - 0.5
    res = np.fromfile(filename, dtype='uint16').reshape(-1, len(channels))
    os.remove(filename)

    if not in_volts:
        return res.transpose()

    out = []
    for i, c in enumerate(channels):
        out.append(res[:, i]*CALIBRATION[c][0] + CALIBRATION[c][1])

    return np.row_stack(out)


if __name__ == '__main__':
    chan = sys.argv[1]

    rm = pyvisa.ResourceManager()
    d = get_data(1, channels=chan)
    tau = 1/250_000_000
    t = np.arange(len(d[0]))*tau

    plt.plot(t, d[0])
    plt.show()
