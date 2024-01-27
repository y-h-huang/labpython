import numpy as np
import os
from collections import defaultdict, namedtuple

dbm_to_mw = lambda x: 10**(x/10)
mw_to_dbm = lambda x: np.log10(x)*10

def ratio_to_db(r, mode='amplitude'):
    if mode == 'amplitude':
        prefactor = 20
    elif mode == 'power':
        prefactor = 10
    else:
        raise ValueError(f'Unknown mode \'{mode}\'; must be \'amplitude\' or \'power\'')

    return prefactor*np.log10(r)

def db_to_ratio(db, mode='amplitude'):
    if mode == 'amplitude':
        prefactor = 20
    elif mode == 'power':
        prefactor = 10
    else:
        raise ValueError(f'Unknown mode \'{mode}\'; must be \'amplitude\' or \'power\'')

    return 10**(db/prefactor)


def linpower(dbm1, dbm2, n, return_dbm=False):
    mws = np.linspace(dbm_to_mw(dbm1), dbm_to_mw(dbm2), n)
    return mw_to_dbm(mws) if return_dbm else mws

def sort_1d_arrays(*arrays):
    '''
    Given a list of arrays, sort all of them according to the order of 
    the first one, then the second one, etc.
    '''
    return [np.array(x) for x in zip(*sorted((x for x in zip(*arrays))))]

def uniques(x):
    '''
    Return sorted unique values of an numpy array
    '''
    return np.array(list(sorted(set(x))))

def funcfunc(x):
    import re

    if isinstance(x, str):
        func = lambda f: f.endswith(x)
    elif isinstance(x, re.Pattern):
        func = lambda f: re.search(x, f) is not None
    elif callable(x):
        func = x
    else:
        raise ValueError('Unknown type to convert to True/False function')

    return func

def find_files(root='.', func=lambda x: True):
    '''
    Recursively walk the file tree and yields each file whose full name
    (not the absolute path, though) causes the func to return True
    '''
    func = funcfunc(func)

    for p, _, fs in os.walk(root):
        for f in fs:
            fn = os.path.join(p, f)
            if func(fn):
                yield fn

def find_dirs(root='.', func=lambda x: True):
    '''
    Recursively walk the file tree and yields each directory whose full name
    (not the absolute path, though) causes the func to return True
    '''
    func = funcfunc(func)

    for p, ds, _ in os.walk(root):
        for d in ds:
            dn = os.path.join(p, d)
            if func(dn):
                yield dn

class VNASweepTrace():
    '''
    Wrapper class for a single VNA sweep.
    '''
    def __init__(self, f, x=None, y=None):
        if x is None:
            try:
                f, x, y = f.transpose()
            except ValueError:
                z = f
                f = z[:, 0].real
                x = z[:, 1:].real
                y = z[:, 1:].imag
        elif y is None:
            z = x
            x, y = z.real, z.imag

        self.f, self.x, self.y = f, x, y
        self.z = self.x + 1j*self.y
        self.R = np.hypot(self.x, self.y)
        self.Rsquared = self.x*self.x + self.y*self.y

class VNASweepData():
    '''
    Wrapper class for a set of VNA sweep data. All hash keys are converted to
    object fields.
    '''
    def __init__(self, fn):
        for n, v in np.load(fn, allow_pickle=True).item().items():
            self.__dict__[n] = VNASweepTrace(v) if n.startswith('data_') else v

    def __getitem__(self, n):
        return self.__dict__[n]

    def __setitem__(self, n, val):
        self.__dict__[n] = val

def add_key_handler(fig, func):
    def callback(e):
        if func(fig, e):
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', callback)

def clustering(x, spacing):
    n = len(x)
    idx = np.arange(0, n)
    _, idx = sort_1d_arrays(x, idx)

    groups = [[idx[0]]]
    for i in range(1, n):
        if x[idx[i]] - x[idx[i - 1]] >= spacing:
            groups.append([])
        groups[-1].append(idx[i])

    return groups
    

def random_element(seq):
    import random

    result = None
    for n, x in enumerate(seq):
        if random.randint(0, n) == n:
            result = x

    return result

def hash_to_tuple(hash, name='_'):
    return namedtuple(name, hash.keys())(*list(hash.values()))

def load_data(filename, tuple_name=None):
    hash = np.load(filename, allow_pickle=True).item()
    if tuple_name is None:
        return hash

    return hash_to_tuple(hash, tuple_name)

def extended_phase(z):
    ang = np.arctan2(z.imag, z.real)
    d = np.diff(ang, prepend=ang[0])
    da = np.zeros_like(d)
    da[d > np.pi] -= 2*np.pi
    da[d < -np.pi] += 2*np.pi
    return ang + np.cumsum(da)

def circle_fit(z):
    import lmfit
    x, y = z.real, z.imag

    def residual(parm):
        x0 = parm['x0'].value
        y0 = parm['y0'].value
        r2 = parm['r2'].value

        return (x - x0)**2 + (y - y0)**2 - r2

    x0 = (x.max() + x.min())/2
    y0 = (y.max() + y.min())/2
    r2 = np.mean(x*x + y*y)

    parm = lmfit.Parameters()
    parm.add('x0', x0)#, min=x0 - dr, max=x0 + dr)
    parm.add('y0', y0)#, min=y0 - dr, max=y0 + dr)
    parm.add('r2', r2)

    mini = lmfit.Minimizer(residual, parm).minimize()
    parm = mini.params
    
    return parm['x0'].value + 1j*parm['y0'].value, parm['r2'].value**.5

def chebpoly(x, y, deg):
    from numpy.polynomial.chebyshev import chebfit, chebval
    return chebval(x, chebfit(x, y, deg=deg))

def weighed_smooth(x, width, weights):
    w = np.pad(weights, width)
    x = np.pad(x, width)

    wsum = np.cumsum(w)
    wsum = wsum[2*width:] - wsum[:-2*width]

    fsum = np.cumsum(w*x)
    fsum = fsum[2*width:] - fsum[:-2*width]

    return fsum/wsum

class AvgStdAccumulator():
    def __init__(self):
        self.clear()

    def clear(self):
        self.set(0, 0, 0)

    def set(self, n, sum, sqsum):
        self.n, self.sum, self.sqsum = n, sum, sqsum
        return self

    def append(self, x):
        self.sum += x
        self.sqsum += np.abs(x)**2
        self.n += 1

    def mean(self):
        return self.sum/self.n

    def std(self):
        return (self.sqsum*self.n - self.sum**2)**.5/self.n

    def meanstd(self):
        return self.mean(), self.std()

    def serialize(self):
        return {'n': self.n, 'sum': self.sum, 'sqsum': self.sqsum}

    def from_dict(self, d):
        return self.set(**d)


class AvgStdAccumulatorC():
    def __init__(self):
        self.clear()

    def clear(self):
        self.set(0, 0, 0, 0, 0)

    def set(self, n, xsum, xsqsum, ysum, ysqsum):
        self.n, self.xsum, self.xsqsum, self.ysum, self.ysqsum = \
            n, xsum, xsqsum, ysum, ysqsum
        return self

    def append(self, z):
        x, y = z.real, z.imag

        self.xsum += x
        self.xsqsum += np.abs(x)**2

        self.ysum += y
        self.ysqsum += np.abs(y)**2
        self.n += 1

    def mean(self):
        z = self.xsum + 1j*self.ysum
        return z/self.n

    def std(self):
        x = (self.xsqsum*self.n - self.xsum**2)**.5/self.n
        y = (self.ysqsum*self.n - self.ysum**2)**.5/self.n
        return x + 1j*y

    def meanstd(self):
        return self.mean(), self.std()

    def serialize(self):
        return { 'n': self.n,
                 'xsum': self.xsum,
                 'ysum': self.ysum,
                 'xsqsum': self.xsqsum,
                 'ysqsum': self.ysqsum}

    def from_dict(self, d):
        return self.set(**d)

def timestamp(t=None):
    import time
    return time.strftime('%Y-%m-%d_%H%M%S')


### def fft(dt, z, with_f=True, with_fft=True):
###     ''' dt: time step; v: complex time domain signal '''
###     res = []
### 
###     if with_f:
###         f_max = 0.5/dt
###         n = len(z)//2
###         res.append(np.arange(-n, n)*(f_max/n))
###         if not with_fft:
###             return res[0]
### 
###     if with_fft:
###         res.append(np.fft.fftshift(np.fft.fft(z)))
###         if not with_f:
###             return res[0]
### 
###     return res

class Attr(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v


if __name__ == '__main__':
    print(clustering(np.array([2, 1, 2, 4, 5, 6, 10, 11, 12]), 2))
