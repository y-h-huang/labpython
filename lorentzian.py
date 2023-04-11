from collections import namedtuple
from itertools import count
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
import lmfit
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from vnaplot import VNA_plot

import matplotlib.pyplot as plt

res_keys = 'amplitude f0 width phase f_offset f_scale poly chi2'.split()
FitResult = namedtuple('LorentzianFit', res_keys)
FitFuncs = namedtuple('LorentzianFuncs', 'resonance wind unwind baseline')


def lorentzian(f, f0, w):
    return 1j*f0*w/((f0 + f)*(f0 - f) + 1j*f*w)


def deslope(f, z):
    l = 30
    zz = np.concatenate((z[:l], z[-l:]))
    ff = np.concatenate((f[:l], f[-l:]))
    p = np.polyfit(ff, zz, 1)

    zz = z - np.polyval(p, f)
    return zz, p


def guess(f, z, smooth_len):

    f_off = f[0]
    Fstar = f_off**2

    zz, pslope = deslope(f - f_off, z)
    smooth = gaussian_filter(zz, smooth_len)
    sabs = np.abs(smooth)
    smax = np.max(sabs)

    peak_sel = sabs > smax*.8
    f0 = np.mean(f[peak_sel])

    sel = (f > f0 - 20) & (f < f0 + 20)
    phase = np.arctan2(np.mean(zz[sel].imag), np.mean(zz[sel].real))

    sel = (f > f0 - 50) & (f < f0 + 50)
    fs = f[sel]
    ss = sabs[sel]

    F = (fs - f_off)*(fs + f_off)
    p = np.polyfit(F, 1/ss**2, 2)
    p2, p1, p0 = p

    F0 = ((p0 - p1*Fstar)/p2 + Fstar**2)**.5
    W = p1/p2 + 2*(F0 - Fstar)
    #w = W**.5 if W > 0 else 1e3
    w = W**.5

    f0 = F0**.5

    amp = 1/(p2*W*F0)**.5

    p = lmfit.Parameters()
    p.add('amplitude', amp)
    p.add('phase', phase)
    p.add('f0', f0)
    p.add('width', w, min=w/10)
    p.add('f_offset', f_off, vary=False)
    p.add('f_scale', f.max() - f.min(), vary=False)

    return p


def get_funcs(res):
    f0 = res.f0[0]
    w = res.width[0]
    amp = res.amplitude[0]*np.exp(1j*res.phase[0])
    f_oft = res.f_offset
    f_scale = res.f_scale
    p = res.poly

    def resonance(f):
        return lorentzian(f, f0, w)

    def baseline(f):
        return chebval((f - f_oft)/f_scale, p)
        #polyval(p, (f - f_oft)/f_scale)

    def wind(f, c):
        return c*amp + baseline(f)

    def unwind(f, z):
        return (z - baseline(f))/amp

    return FitFuncs(resonance, wind, unwind, baseline)


def fit(f, z, *, smooth=20, deg=1, **guesses):
    p = guess(f, z, smooth_len=smooth)

    f_oft = p['f_offset'].value
    f_scale = p['f_scale'].value

    df = f - f_oft
    dfu = df/f_scale

    for k, v in guesses.items():
        if k in p:
            if k == 'width':
                p.add(k, v, min=0)
            elif k == 'f_scale' or k == 'f_offset':
                p.add(k, v, vary=False)
            else:
                p.add(k, v)

    def pre_fit(parm):
        phase = np.exp(1j*parm['phase'].value)
        amp = parm['amplitude'].value*phase
        w = parm['width'].value
        f0 = parm['f0'].value

        L = lorentzian(f, f0, w)
        poly = chebfit(dfu, z - L*amp, deg=deg)
        p = chebval(dfu, poly)
        return phase, w, f0, amp, L, poly, p

    def residual(parm):
        _, _, _, amp, L, poly, p = pre_fit(parm)

        diff = (L*amp + p - z).view(type(z[0].real))

        if 0:
            print()
            for k in parm.keys():
                print(k, parm[k].value)
            print(np.sum(diff**2))

            vp = VNA_plot(figsize=(10, 4))
            vp.plot(f, z)
            vp.plot(f, L*amp + p)
            vp.show()
    
        return diff

    mini = lmfit.Minimizer(residual, p).minimize()
    parm = mini.params

    amp, f0, w, phase = [(parm[k].value, parm[k].stderr) for k in 'amplitude f0 width phase'.split()]
    poly = pre_fit(parm)[-2]
    return FitResult(amp, f0, w, phase, f_oft, f_scale, poly, mini.chisqr)


def main():
    from utils import load_data, find_files

    d = load_data('lorentzian_test.npy', 'data')
    f = d.f
    z = d.z

    res = fit(f, z, deg=3, smooth=20)
    funcs = get_funcs(res)

    vp = VNA_plot()
    if 0:
        vp.plot(f, z)
        vp.plot(f, funcs.wind(f, funcs.resonance(f)))
        vp.plot(f, funcs.baseline(f))
    else:
        vp.plot(f, funcs.unwind(f, z))
        vp.plot(f, funcs.resonance(f))
        vp.plot(f, funcs.unwind(f, funcs.baseline(f)))
    vp.show()
    
if __name__ == '__main__':
    main()
