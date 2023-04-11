import numpy as np
from utils import chebpoly, extended_phase
from collections import namedtuple
from scipy.ndimage import gaussian_filter
from numpy.polynomial.chebyshev import chebfit, chebval
import lmfit

res_keys = 'f_a kappa_a kappa_ext mismatch electrical_delay f_offset f_scale poly chi2'.split()
FitResult = namedtuple('CavityFit', res_keys)
FitFuncs = namedtuple('CavityFuncs', 'resonance wind unwind baseline')

def resonance_shape(f, fa, mis, ka, kext):
    return 1 - np.exp(1j*mis)*kext/(1j*(f - fa) + ka/2)

def find_runs(x):
    return np.split(np.arange(0, len(x)), np.where(np.diff(x) != 0)[0] + 1)

def guess(f, z, width):

    df = f - f[0]
    fscale = f.max() - f.min()

    zs = gaussian_filter(z, 10)

    ph = extended_phase(zs)
    delay, phase = np.polyfit(df, ph, 1)

    zs *= np.exp(-1j*(delay*df + phase))

#    VNA_plot().plot(f, zs).show()

    za = np.abs(zs)

    zmax = np.max(za)
    zmin = np.min(za)

    sel = (za - zmin) < (zmax - zmin)*.1
    f0 = np.mean(f[sel])

    ploss = np.polyfit((df[0], df[-1]), (za[0], za[-1]), deg=1)
    zinv = 1 - zs/np.polyval(ploss, df)

    za = np.abs(zinv)
    zmax = np.max(zinv)
    sel = za > zmax*.9

    x = (f[sel] - f0)**2
    y = 1/za[sel]**2

    p1, p0 = np.polyfit(x, y, 1)
    ke = 1/p1**0.5
    ka = (p0/p1*4)**.5

    p = lmfit.Parameters()
    p.add('kappa_a', ka, min=ka/10)
    p.add('kappa_ext', ke, min=ke/10)
    p.add('mismatch', 0.05, min=0, max=0.1)
    p.add('electrical_delay', delay, min=delay - 1e-8, max=delay + 1e-8)
    p.add('f_a', f0)
    p.add('f_offset', f.min(), vary=False)
    p.add('f_scale', fscale, vary=False)
    return p


def get_funcs(r):
    fa = r.f_a[0]
    mis = r.mismatch[0]
    ka = r.kappa_a[0]
    kext = r.kappa_ext[0]
    delay = r.electrical_delay[0]
    f0 = r.f_offset
    fs = r.f_scale
    p = r.poly

    def resonance(f):
        return resonance_shape(f, fa, mis, ka, kext)

    def scaling(f):
        df = f - f0
        return np.exp(1j*delay*df)*chebval(df/fs, p)
        
    def wind(f, z):
        return z*scaling(f)

    def unwind(f, z):
        return z/scaling(f)

    return FitFuncs(resonance, wind, unwind, scaling)

def fit(f, z, deg=0, maxiter=None, debug_plot=False, **guesses):
    if 'kappa_a' in guesses:
        width = guesses['kappa_a']/3
    else:
        width = (f.max() - f.min())/50

    parm = guess(f, z, width=width)

    for k, v in guesses.items():
        if k in parm:
            #parm.add(k, v, vary=k != 'electrical_delay' and k != 'f_offset')
            if k.startswith('kappa_') or k.startswith('gamma_'):
                min_val = v/10
            else:
                min_val = None
            parm.add(k, v, min=min_val, vary=k != 'f_offset')

    f0 = parm['f_offset'].value
    fs = parm['f_scale'].value
    df = f - f0
    df_u = df/fs

    vp = None

    def pre_fit(parm):
        f_a = parm['f_a'].value
        kappa_a = parm['kappa_a'].value
        kappa_ext = parm['kappa_ext'].value
        mismatch = parm['mismatch'].value
        delay = parm['electrical_delay'].value
        phase = np.exp(1j*delay*df)
        C = resonance_shape(f, f_a, mismatch, kappa_a, kappa_ext)
        poly = chebfit(df_u, z/(C*phase), deg=deg)
        p = chebval(df_u, poly)*phase

        return C, poly, p

    def residual(parm):
        C, poly, p = pre_fit(parm)

        if debug_plot:
            from vnaplot import VNA_plot
            for k in parm.keys():
                print(k, parm[k].value)
            print()
            nonlocal vp
            if vp is None:
                vp = VNA_plot(figsize=(10, 4)).show(block=False)
            vp.clear()
            vp.plot(f, z)
            vp.plot(f, C)
            vp.draw()

        return (C - z/p).view(type(z[0].real))

    mini = lmfit.Minimizer(residual, parm).minimize(max_nfev=maxiter)
    p = mini.params

    C, poly, _ = pre_fit(p)

    return FitResult(*[ (p[k].value, p[k].stderr) for k in res_keys[:5] ], f0, fs, poly, mini.chisqr)


if __name__ == '__main__':
    import os
    from utils import load_data, find_files
    import random
    from vnaplot import VNA_plot

    filename = random.choice(list(find_files('../200micron_yig/data_2022-11-07/bias_sweep', '.npy')))
    #filename = '../200micron_yig/data_2022-11-07/bias_sweep/20221107-12_32_53.npy'
    #for filename in find_files('../200micron_yig/data_2022-11-07/bias_sweep', '.npy'):
    #for filename in ['../200micron_yig/data_2022-11-07/bias_sweep/20221107-13_02_37.npy']:
    for filename in ['cavity_test.npy']:
        print(filename)
        d = load_data(filename, 'D')

        f, z = d.f, d.z
        #idx = (d.f > -7.29e9) & (d.f < 7.37e9 + 1e10)
        #f, z = d.f[idx], d.z[idx]

        res = fit(f, z, deg=5, debug_plot=False)
        print(res)

        funcs = get_funcs(res)
        zfit = funcs.resonance(f)

        vp = VNA_plot()
        vp.plot(f, z)
        vp.plot(f, funcs.wind(f, zfit))
        vp.plot(f, funcs.baseline(f))
        vp.show()
