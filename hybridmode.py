import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.ndimage import gaussian_filter
from scipy.signal import deconvolve
import lmfit
from utils import extended_phase, chebpoly
from numpy.polynomial.chebyshev import chebfit, chebval
from vnaplot import VNA_plot

res_keys = 'f_a kappa_a f_m gamma_m g_am kappa_ext mismatch electrical_delay f_offset f_scale poly chi2'.split()
FitResult = namedtuple('HybridFit', res_keys)
FitFuncs = namedtuple('HybridFuncs', 'resonance wind unwind peaks baseline')

def resonance_shape(f, fa, fm, ka, km, gam, kext, mismatch):
    return 1 - np.exp(1j*mismatch)*kext/(1j*(f - fa) + ka/2 + gam**2/(1j*(f - fm) + km/2))


def find_runs(x):
    return np.split(np.arange(0, len(x)), np.where(np.diff(x) != 0)[0] + 1)

# width gives rough estimate of peak widths in Hz
def guess(f, z, width=1e6, deg=0, peak_only=False):
    d = 30  # number of points at either end to fit for amp/phase correction
    delta_f = np.min(np.diff(f))
    
    smooth_width = 5
    zz = gaussian_filter(z, smooth_width, mode='nearest')

    angle = extended_phase(zz)
    da = np.diff(angle, prepend=angle[0])

    w = da - chebpoly(f, da, deg=1)
    smooth_len = int(width/delta_f/2)
    ws = gaussian_filter(w, smooth_len)
    ws[:smooth_len*2] = 0
    ws[-2*smooth_len:] = 0

    baseline = ws.max()/10

    runs = find_runs(ws > baseline)[1::2]
    indices = sorted([(len(x), i, x) for i, x in enumerate(runs)], reverse=True)[:2]
    runs = [x for _, _, x in indices]

    if runs[0][0] > runs[1][0]:
        runs = [runs[1], runs[0]]

    peak_minus = np.mean(f[runs[0]])
    peak_plus = np.mean(f[runs[1]])

    if peak_only:
        return peak_minus, peak_plus

    import cavity
    sel = (f > peak_minus - width*5) & (f < peak_minus + width*3)
    fit_minus = cavity.fit(f[sel], z[sel], deg=1, maxiter=100)
    kminus = fit_minus.kappa_a[0]
    keminus = fit_minus.kappa_ext[0]
    fminus = fit_minus.f_a[0]

    sel = (f > peak_plus - width) & (f < peak_plus + width)
    fit_plus = cavity.fit(f[sel], z[sel], deg=1, maxiter=100)
    kplus = fit_plus.kappa_a[0]
    keplus = fit_plus.kappa_ext[0]
    fplus = fit_plus.f_a[0]

    kext = keplus + keminus
    c2t = (keminus - keplus)/kext # cos(2 theta)
    
    #print(kplus, kminus, kext, c2t)

    ka = (kplus*(1 - 1/c2t) + kminus*(1 + 1/c2t))/2
    km = (kplus*(1 + 1/c2t) + kminus*(1 - 1/c2t))/2
    if km < 0:
        km = -km
    fm = (fplus*(1 + c2t) + fminus*(1 - c2t))/2
    fa = fplus + fminus - fm
    gam = ((fplus - fminus)**2 - (fm - fa)**2)**.5/2

    f0 = f.min()
    f1 = f.max()
    fscale = f1 - f0

    ff = np.concatenate((f[:d], f[-d:]))
    zz = np.concatenate((z[:d], z[-d:]))
    ff = (ff - f0)/fscale

    p = lmfit.Parameters()
    p.add('f_offset', f0, vary=False)
    p.add('f_scale', f1 - f0, vary=False)
    p.add('f_a', fa, min=fa - gam/10, max=fa + gam/10)
    p.add('f_m', fm, min=fm - gam/10, max=fm + gam/10)
    p.add('kappa_a', ka, min=ka/2, max=ka*2)
    p.add('gamma_m', km, min=km/2, max=km*2)
    p.add('g_am', gam, min=1e5)
    p.add('mismatch', 0.05, min=-.5, max=.5, vary=True)
    p.add('kappa_ext', kext, min=kext/10, max=kext*5)

    pp = np.polyfit(f, angle, deg=1)
    p.add('electrical_delay', pp[0], min=-1e-6, max=1e-6)

    return p

def wild_guess(f):
    #print('Wild guess for hybrid mode')

    f0 = f.min()
    f1 = f.max()
    fscale = f1 - f0
    fmid = (f0 + f1)/2

    p = lmfit.Parameters()
    p.add('f_offset', f0, vary=False)
    p.add('f_scale', f1 - f0, vary=False)
    p.add('f_a', fmid, min=f0, max=f1)
    p.add('f_m', fmid, min=f0, max=f1)
    p.add('kappa_a', 1e6, min=1e5, max=10e6)
    p.add('gamma_m', 2e6, min=5e5, max=20e6)
    p.add('g_am', 2e6, min=1e6, max=20e6)
    p.add('mismatch', 0.05, min=-.5, max=.5)
    p.add('kappa_ext', 1e6, min=1e5, max=5e6)
    p.add('electrical_delay', 0, min=-1e-6, max=1e-6)

    return p

def get_funcs(r):
    fa = r.f_a[0]
    mis = r.mismatch[0]
    fm = r.f_m[0]
    ka = r.kappa_a[0]
    km = r.gamma_m[0]
    kext = r.kappa_ext[0]
    delay = r.electrical_delay[0]
    f0 = r.f_offset
    fscale = r.f_scale
    gam = r.g_am[0]
    poly = r.poly

    def resonance(f):
        return resonance_shape(f, fa, fm, ka, km, gam, kext, mis)

    def scaling(f):
        df = f - f0
        return np.exp(1j*delay*df)*chebval(df/fscale, poly)

    def wind(f, z):
        return z*scaling(f)

    def unwind(f, z):
        return z/scaling(f)

    def peaks():
        sam = (fa + fm)/2
        dam = (fa - fm)/2
        delta = (dam**2 + gam**2)**.5
        return sam - delta, sam + delta

    return FitFuncs(resonance, wind, unwind, peaks, scaling)

def fit(f, z, *, prev=None, rf_freq=None, deg=0, maxiter=1000, weights=None, **guesses):

    if prev is not None:
        for k in res_keys[:8]:
            if k not in guesses:
                guesses[k] = getattr(prev, k)[0]

    if 'kappa_a' in guesses:
        width = guesses['kappa_a']*2
    else:
        width = (f.max() - f.min())/50

    try:
        parm = guess(f, z, width=width, deg=deg)
    except IndexError:
        parm = wild_guess(f)

    for k in guesses.keys():
        if k.startswith('kappa_') or k.startswith('gamma_') or k.startswith('f_'):
            parm.add(k, guesses[k], min=0)
        else:
            parm.add(k, guesses[k])

    #print(parm)
    f0 = parm['f_offset'].value
    fscale = parm['f_scale'].value

    df = f - f0
    df_norm = df/fscale

    #weights = None
    if rf_freq is not None:
        weights = np.ones_like(f)
        weights[(f > rf_freq - 5.5e6) & (f < rf_freq + 5e5)] = 0

    def pre_fit(parm):
        f_a = parm['f_a'].value
        f_m = parm['f_m'].value
        gamma_m = parm['gamma_m'].value
        kappa_a = parm['kappa_a'].value
        g_am = parm['g_am'].value
        kappa_ext = parm['kappa_ext'].value
        mismatch = parm['mismatch'].value
        delay = parm['electrical_delay'].value
        phase = np.exp(1j*delay*df)
        C = resonance_shape(f, f_a, f_m, kappa_a, gamma_m, g_am, kappa_ext, mismatch)
        poly = chebfit(df_norm, z/(C*phase), deg=deg)
        p = chebval(df_norm, poly)*phase

        return C, poly, p

    vp = None
    def residual(parm):
        C, poly, p = pre_fit(parm)

        if 0:
            nonlocal vp

            if vp is None:
                vp = VNA_plot()
                vp.show(block=False)

            vp.clear()
            vp.plot(f, z/p)
            vp.plot(f, C)
            vp.draw()
        
        diff = C*p - z
        if weights is not None:
            diff *= weights
        return diff.view(type(z[0].real))

    mini = lmfit.Minimizer(residual, parm).minimize(max_nfev=maxiter)
    parm = mini.params

    C, poly, _ = pre_fit(parm)

    return FitResult(*[ (parm[k].value, parm[k].stderr) for k in res_keys[:8] ], f0, fscale, poly, mini.chisqr)

if __name__ == '__main__':
    from utils import load_data
    from vnaplot import VNA_plot

    d = load_data('trace.npy', 'Data')
    f, z = d.f, d.z
    z = np.exp(1j*0.204e-6*(f - f[0]))*z
    #vp = VNA_plot().plot(d.f, d.z)
    #vp.draw()
    #vp.show()

    res = fit(f, z, deg=3, kappa_a=1e6)
    print(res)
    fun = get_funcs(res)

    #z = fun.unwind(d.f, d.z)
    #zfit = fun.resonance(d.f)
    zfit = fun.wind(f, fun.resonance(f))
    VNA_plot().plot(f, z).plot(f, zfit).plot(f, fun.baseline(f)).show()


