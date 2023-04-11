import matplotlib.pyplot as plt
import numpy as np

class VNA_plot():
    def __init__(self, fig=None, use_db=False, xy_origin=True, phase_plot=False, **fig_args):
        plt.ioff()
        options = {'figsize': (10, 4)}
        options.update(fig_args)

        if fig is None:
            fig = plt.figure(**options)

        self.fig = f = fig
        #self.fig._zidx = 0
        self.xy_origin = xy_origin
        self.phase_plot = phase_plot

        self.setup_axes()
        self.use_db = use_db

    def show_phase(self, show):
        self.phase_plot = show
        self.setup_axes()
        return self
    
    def show_xy_origin(self, show):
        self.xy_origin = show
        self.setup_axes()
        return self

    def setup_axes(self):
        f = self.fig
        f.clear()

        axes = f.subplots(2 if self.phase_plot else 3, 2, squeeze=False, sharex='col')
        axes[0][1].remove()
        axes[1][1].remove()

        if not self.phase_plot:
            axes[2][1].remove()

        self.xy = f.add_subplot(3, 2, (2, 6), sharex=None)
        self.xy.grid(1)

        if not self.phase_plot:
            self.x = axes[0][0]
            self.y = axes[1][0]
            self.r = axes[2][0]
            self.p = None
        else:
            self.p = axes[0][0]
            self.r = axes[1][0]
            self.x = self.y = None

        self.xy.set_aspect(1)

    def extended_phase(self, z):
        ang = np.arctan2(z.imag, z.real)
        d = np.diff(ang, prepend=ang[0])
        da = np.zeros_like(d)
        da[d > np.pi] -= 2*np.pi
        da[d < -np.pi] += 2*np.pi
        return ang + np.cumsum(da)

    def vline(self, x, **kwargs):
        for ax in (self.x, self.y, self.r):
            ax.axvline(x, **kwargs)
        return self

    def clear(self):
        for ax in [self.p, self.r, self.xy] if self.phase_plot else [self.x, self.y, self.r, self.xy]:
            ax.clear()
        if self.phase_plot:
            self.p.set_ylabel('Phase/$\\pi$')
        else:
            self.x.set_ylabel('X')
            self.y.set_ylabel('Y')

        self.r.set_ylabel('R (dB)' if self.use_db else 'R')
        self.r.set_xlabel('Frequency')

        return self

    def show_db(self, db):
        self.use_db = db
        return self

    def plot(self, f, z, **kwargs):
        if not self.phase_plot:
            self.x.plot(f, z.real, **kwargs)
            self.y.plot(f, z.imag, **kwargs)
        else:
            self.p.plot(f, self.extended_phase(z)/np.pi, **kwargs)

        if self.use_db:
            db = 20*np.log10(np.abs(z))
            self.r.plot(f, db, **kwargs)
        else:
            self.r.plot(f, np.abs(z), **kwargs)

        self.xy.plot(z.real, z.imag, **kwargs)

        if self.xy_origin:
            r = np.max(np.abs(z)) * 1.1
            x0, x1 = self.xy.get_xlim()
            y0, y1 = self.xy.get_ylim()
            x0 = min(x0, -r)
            x1 = max(x1, r)
            y0 = min(y0, -r)
            y1 = max(y1, r)

            self.xy.set_xlim([x0, x1])
            self.xy.set_ylim([y0, y1])

        return self

    def add_key_handler(self, handler):
        def onkey(e):
            k = e.key
            if handler(k):
                self.draw()

        self.fig.canvas.mpl_connect('key_press_event', onkey)

    def finalize(self):
        if self.xy_origin:
            self.xy.axhline(0, lw=0.25, color=[.6, 0, 0])
            self.xy.axvline(0, lw=0.25, color=[.6, 0, 0])

    def show(self, **kwargs):
        self.finalize()
        plt.show(**kwargs)
        return self

    def draw(self):
        self.finalize()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
