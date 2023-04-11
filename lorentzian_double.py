import lorentzian as L
from vnaplot import VNA_plot
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
import lmfit
from collections import namedtuple

res_keys = 'amplitude1 f1 width1 phase1 amplitude2 f2 width2 phase2 f_offset f_scale poly chi2'.split()
FitResult = namedtuple('DoubleLorentzianFit', res_keys)
FitFuncs = namedtuple('DoubleLorentzianFuncs', 'resonance wind unwind baseline')

def get_funcs(res):
    f1 = res.f1[0]
    w1 = res.width1[0]
    amp1 = res.amplitude1[0]*np.exp(1j*res.phase1[0])

    f2 = res.f2[0]
    w2 = res.width2[0]
    amp2 = res.amplitude2[0]*np.exp(1j*res.phase2[0])

    f_oft = res.f_offset
    f_scale = res.f_scale
    p = res.poly

    def resonance(f):
        l1 = L.lorentzian(f, f1, w1)
        l2 = L.lorentzian(f, f2, w2)

        return l1 + l2*(amp2/amp1)

    def baseline(f):
        return chebval((f - f_oft)/f_scale, p)

    def wind(f, z):
        return z*amp1 + baseline(f)

    def unwind(f, z):
        return (z - baseline(f))/amp1
    
    return FitFuncs(resonance, wind, unwind, baseline)

def fit(f, z, deg=0):
    fit1 = L.fit(f, z, deg=deg)
    fun = L.get_funcs(fit1)
    zfit = fun.wind(f, fun.resonance(f))
    fit2 = L.fit(f, z - zfit, deg=0)
    fun2 = L.get_funcs(fit2)
    zfit2 = fun2.wind(f, fun2.resonance(f))

    p = lmfit.Parameters()
    p.add('f1', fit1.f0[0])
    p.add('width1', fit1.width[0])
    p.add('phase1', fit1.phase[0])
    p.add('amplitude1', fit1.amplitude[0])

    p.add('f2', fit2.f0[0])
    p.add('width2', fit2.width[0])
    p.add('phase2', fit2.phase[0])
    p.add('amplitude2', fit2.amplitude[0])

    p.add('f_offset', fit1.f_offset, vary=False)
    p.add('f_scale', fit1.f_scale, vary=False)

    df = f - fit1.f_offset
    dfu = df/fit1.f_scale

    def pre_fit(parm):
        phase1 = np.exp(1j*parm['phase1'].value)
        amp1 = parm['amplitude1'].value*phase1
        w1 = parm['width1'].value
        f1 = parm['f1'].value

        phase2 = np.exp(1j*parm['phase2'].value)
        amp2 = parm['amplitude2'].value*phase2
        w2 = parm['width2'].value
        f2 = parm['f2'].value

        l1 = L.lorentzian(f, f1, w1)
        l2 = L.lorentzian(f, f2, w2)

        zz = l1 + l2*(amp2/amp1)
        poly = chebfit(dfu, z - zz*amp1, deg=deg)
        p = chebval(dfu, poly)

        return amp1, zz, poly, p

    def residual(parm):
        amp, zz, poly, p = pre_fit(parm)
        diff = (zz*amp + p - z).view(type(z[0].real))

        return diff

    mini = lmfit.Minimizer(residual, p).minimize()
    parm = mini.params

    amp1, f1, w1, phase1 = [(parm[k].value, parm[k].stderr) for k in \
                            'amplitude1 f1 width1 phase1'.split()]

    amp2, f2, w2, phase2 = [(parm[k].value, parm[k].stderr) for k in \
                            'amplitude2 f2 width2 phase2'.split()]

    poly = pre_fit(parm)[-2]

    return FitResult(*[[parm[k].value, parm[k].stderr] for k in res_keys[:8]],
                     fit1.f_offset, fit1.f_scale, poly, mini.chisqr)


if __name__ == '__main__':
    def read_data(filename):
        with open(filename, 'rt') as f:
            f, _, x, y = [np.array(x) for x in zip(*[[float(t) for t in l.split(';')] for l in list(f)[7:]])]
        return f, x + 1j*y

    _, z0 = read_data('cap_BG_0V.txt')
    f, z = read_data('cap_BG_10V.txt')

    ft = fit(f, z - z0, deg=1)

    for k in ft._fields:
        print(f'{k:12s}', getattr(ft, k))

    fun = get_funcs(ft)
    zfit = fun.wind(f, fun.resonance(f))

    VNA_plot().plot(f, z - z0).plot(f, zfit).show()
