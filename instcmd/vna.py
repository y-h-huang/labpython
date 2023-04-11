import numpy as np
from instcmd.instr_cmd import DeviceCommandLine
from vnaplot import VNA_plot
from collections import defaultdict

class cmd_VNA(DeviceCommandLine):
    devname = 'cmd_VNA'

    def __init__(self, *args):
        super().__init__(*args)
        self.vp = VNA_plot(figsize=(10, 4), phase_plot=True)

        def key_handle(k):
            if k == 'p':
                self.vp.show_phase(not self.vp.phase_plot)
            elif k == 'o':
                self.vp.show_xy_origin(not self.vp.xy_origin)
            else:
                return False
            
            return True
        
        self.vp.add_key_handler(key_handle)
        self.vp.show(block=False)

        self.markers = {}
        self.marker_master = defaultdict(set)
        self.marker_slave = {}

        self.saved_states = {}
        self.freq = self.z = None
        self.linewidth = 1
        self.phase = 0
        self.auto_center = None
        self.loss = None
        self.loss_correction = False

        self.delay = 0
        self.delay_offset = None

    def sweep(self):
        freq, z = self.dev.sweep_current_setting()
        if not self.delay:
            return freq, z

        f0 = self.delay_offset
        return freq, z*np.exp(-1j*(self.phase + self.delay*(freq - f0)))
    
    def _cmd_db_unit(self, *args):
        val = self.parse_bool(args[0])
        self.vp.show_db(val)

    def _cmd_lw(self, linewidth):
        self.linewidth = float(linewidth)
        #self._cmd_update(self.freq, self.z)

    def _cmd_delay(self, *args):
        if args:
            delay, rel = self.parse_number(args[0])
            if rel:
                delay += self.delay

            self.delay = delay
        
        self.delay_offset, _ = self.dev.frequency_range()
        return self.delay
    
    def _cmd_toggle(self, *args):
        cmd = args[0].lower()

        if 'phase'.startswith(cmd):
            self.vp.show_phase(not self.vp.phase_plot)
        elif 'origin'.startswith(cmd):
            self.vp.show_xy_origin(not self.vp.xy_origin)
        else:
            raise ValueError(f'Unknown toggle `{cmd}\'')

        self._cmd_update(self.freq, self.z)

    def _cmd_autocenter(self, *args):
        if not args:
            return self.auto_center

        m = args[0]
        if m.lower() == 'off':
            self.auto_center = None
        else:
            self.auto_center = m

        self._cmd_span('center', str(self.markers[m]))

    _cmd_ac = _cmd_autocenter

    def _cmd_std(self, *args):
        import time, sys
        from utils import AvgStdAccumulatorC

        update_timer = 3
        n = None

        if len(args) >= 1:
            n = int(args[0]) or None

        if len(args) >= 2:
            update_timer = float(args[1])

        avg = AvgStdAccumulatorC()
        print('Averaging indefinitely' if n is None else f'Averaging {n} times')

        t0 = time.time()

        while True:
            freq, z = self.sweep()
            avg.append(z)
            print(f'\r{avg.n}' if n is None else f'\r{avg.n}/{n}', end='')
            sys.stdout.flush()

            t1 = time.time()
            if t1 - t0 > update_timer:
                self._cmd_update(freq, avg.mean(), redraw=False)
                self.vp.plot(freq, avg.std(), lw=self.linewidth).draw()
                t0 = t1

            if n is not None and avg.n >= n:
                break

        print('\r               \r', end='')
        self._cmd_update(freq, avg.mean(), redraw=False)
        self.vp.plot(freq, avg.std(), lw=self.linewidth).draw()


    def _cmd_average(self, *args):
        import time, sys
        from utils import AvgStdAccumulatorC

        update_timer = 3
        n = None

        if len(args) >= 1:
            n = int(args[0]) or None

        if len(args) >= 2:
            update_timer = float(args[1])

        avg = AvgStdAccumulatorC()
        print('Averaging indefinitely' if n is None else f'Averaging {n} times')

        t0 = time.time()

        while True:
            freq, z = self.sweep()
            avg.append(z)
            print(f'\r{avg.n}' if n is None else f'\r{avg.n}/{n}', end='')
            sys.stdout.flush()

            t1 = time.time()
            if t1 - t0 > update_timer:
                self._cmd_update(freq, avg.mean())
                t0 = t1

            if n is not None and avg.n >= n:
                break

        print('\r               \r', end='')
        self._cmd_update(freq, avg.mean())

    _cmd_avg = _cmd_average

    def _cmd_update(self, freq=None, z=None, redraw=True):
        if freq is None:
            freq, z = self.sweep()

        self.freq, self.z = freq, z
        self.vp.clear()

        if self.loss is not None and self.loss_correction:
            z = (z - self.loss[0])/self.loss[1]

        self.vp.plot(freq, z, lw=self.linewidth)

        for ax in self.vp.x, self.vp.y, self.vp.p, self.vp.r:
            if ax:
                a, b = ax.get_ylim()
                for m, f in self.markers.items():
                    if freq[0] <= f <= freq[-1]:
                        ax.text(f, b, m)
                        ax.axvline(f, linestyle='--', color='orange', lw=self.linewidth/2)

        if redraw:
            self.vp.draw()

    def set_marker(self, m, f):
        self.markers[m] = f

        if m in self.marker_master:
            for s in self.marker_master[m]:
                offset = self.marker_slave[s][1]
                self.markers[s] = f + offset

        if self.auto_center == m:
            self._cmd_span('center', str(self.markers[m]))

        return f

    def _cmd_marker(self, *args):
        if not args:
            fref = None
            if 'r' in self.markers:
                fref = self.markers['r']
                print(f'r {int(fref):10d}          0')

            for m, v in sorted(self.markers.items()):
                if m == 'r': continue
                if fref is not None:
                    print(f'{m} {int(v):10d} {int(v - fref):10d}')
                else:
                    print(f'{m} {int(v):10d}')
            return

        args = list(args)
        m = args.pop(0)

        # no more args, return marker status
        if len(args) == 0:
            if m in self.markers:
                return self.markers[m]
            else:
                return 'Off'

        cc = args.pop(0)
        c = cc.lower()

        if c == 'on':
            if m not in self.markers:
                f0, f1 = self.dev.frequency_range()
                f = (f0 + f1)/2
                return self.set_marker(m, f)

            return self.markers[m]

        elif c == 'off':
            del self.markers[m]
            return ''

        elif 'center'.startswith(c):
            return self._cmd_span('center', str(self.markers[m]))

        elif 'pick'.startswith(c):
            f = self.vp.fig.ginput(n=1)[0][0]

        elif 'lock'.startswith(c):
            slave = m
            master = args.pop(0)

            if slave == master:
                raise ValueError(f'Cannot lock marker {slave} to {master}')

            offset = self.markers[slave] - self.markers[master]

            self._cmd_marker(m, 'unlock')

            self.marker_master[master].add(slave)
            self.marker_slave[slave] = (master, offset)

            return self.set_marker(slave, self.markers[master] + offset)

        elif 'unlock'.startswith(c):
            if m in self.marker_slave:
                master = self.marker_slave[m][0]

                self.marker_master[master].discard(m)
                del self.marker_slave[m]

            return self.markers[m]

        else:
            if len(args): # relative to another marker
                f = self.markers[c] + self.parse_number(args.pop(0))[0]

            else:
                f, rel = self.parse_number(cc)
                if rel:
                    if m not in self.markers:
                        f0, f1 = self.dev.frequency_range()
                        f = (f0 + f1)/2
                    else:
                        f += self.markers[m]

        return self.set_marker(m, f)


    def _cmd_autoscale(self):
        self.device_command('autoscale')

    def _cmd_loss(self, *args):
        if not args:
            return 'On' if self.loss_correction else 'Off', self.loss

        cmd = args[0].lower()
        if cmd == 'none':
            self.loss = None
            self.loss_correction = False
            return

        if cmd == 'set':
            from utils import circle_fit

            if len(args) > 1: # load from file
                from utils import load_data
                d = load_data(args[1], 'Loss')
                self.f, self.z = d.f, d.z

            elif self.z is None:
                self._cmd_update()

            self.loss = circle_fit(self.z)
            return self.loss

        self.loss_correction = self.parse_bool(cmd)
        return self._cmd_loss()

    def _cmd_power(self, *args):
        return self.device_command('power', *args)

    def _cmd_output(self, *args):
        return self.dev.output(*[self.parse_bool(x) for x in args])

    def _cmd_bw(self, *args):
        return self.device_command('bandwidth', *args)

    def _cmd_fstart(self, *args):
        return self.device_command('frequency_start', *args)

    def _cmd_fstop(self, *args):
        return self.device_command('frequency_stop', *args)

    def _cmd_measure(self, *args):
        return self.dev.measure(*args)

    def _cmd_span(self, *args):
        f0, f1 = self.dev.frequency_range()
        if not args:
            return f0, f1, f1 - f0

        if 'center'.startswith(args[0]):
            if len(args) == 1:
                return (f0 + f1)/2

            if len(args) != 2:
                raise ValueError('Invalid command: ' + ' '.join(args))

            val, rel = self.parse_number(args[1])
            if rel:
                f0 += val
                f1 += val
            else:
                d = (f1 - f0)/2
                f0 = val - d
                f1 = val + d

        else:
            if len(args) == 1:
                val, rel = self.parse_number(args[0])
                val /= 2
                if rel:
                    f0 -= val
                    f1 += val
                else:
                    f = (f0 + f1)/2
                    f0 = f - val
                    f1 = f + val
            elif len(args) == 2:
                val, rel = self.parse_number(args[0])
                if rel:
                    f0 += val
                else:
                    f0 = val
                val, rel = self.parse_number(args[1])
                if rel:
                    f1 += val
                else:
                    f1 = val
            else:
                raise ValueError(f'bad arguments ' + ' '.join(args))

        self.dev.frequency_range([f0, f1])
        f0, f1 = self.dev.frequency_range()
        return f0, f1, f1 - f0


    def _cmd_continuous(self, *args):
        while True:
            self._cmd_update()

    def _cmd_fit(self, *args):
        op = args[0].lower()
        f, z = self.freq, self.z

        if self.loss_correction and self.loss is not None:
            z = (z - self.loss[0])/self.loss[1]

        options = {}
        for s in args[1:]:
            a, b = s.split('=')
            if a in 'deg maxiter'.split():
                b = int(b)
            else:
                b = float(b)
            options[a] = b

        if 'hybrid'.startswith(op):
            import hybridmode
            fit = hybridmode.fit(f, z, **options)
            fun = hybridmode.get_funcs(fit)
            p0, p1 = fun.peaks()
            print(f'Peak separation {(p1 - p0)/1e6} MHz')

        elif 'cavity'.startswith(op):
            import cavity
            fit = cavity.fit(f, z, **options)
            fun = cavity.get_funcs(fit)

        elif 'lorentzian'.startswith(op):
            import lorentzian
            fit = lorentzian.fit(f, z, **options)
            fun = lorentzian.get_funcs(fit)

        elif 'double_lorentzian'.startswith(op):
            import lorentzian_double as ld
            fit = ld.fit(f, z, **options)
            fun = ld.get_funcs(fit)

        for k in fit._fields:
            print(f'{k:16s} {getattr(fit, k)}')

        #self._cmd_update(f, fun.unwind(f, z), keep=False)
        self.vp.clear()
        zfit = fun.resonance(f)

        self.vp.plot(f, z).plot(f, fun.wind(f, zfit), lw=self.linewidth).plot(f, fun.baseline(f), lw=self.linewidth)
        #self.vp.plot(f, fun.unwind(f, z)).plot(f, zfit).plot(f, fun.baseline(f))
        self.vp.draw()

    def _cmd_state(self, *args):
        def show_state(st):
            for k, v in st.items():
                print('   ', k, v)

        if not args:
            print('Saved VNA configs:')
            for k, v in self.saved_states.items():
                print(k)
                show_state(v)
            return

        cmd = args[0]
        args = args[1:]

        if 'save'.startswith(cmd):
            name = args[0]
            st = {}
            st['power'] = self.dev.power()
            st['freq'] = self.dev.frequency_range()
            st['npts'] = self.dev.num_points()
            st['bw'] = self.dev.bandwidth()
            st['meas'] = self.dev.measure()
            st['markers'] = self.markers.copy()

            self.saved_states[name] = st
            print('saved config')
            show_state(st)

        elif 'load'.startswith(cmd):
            name = args[0]
            st = self.saved_states[name]
            self.dev.power(st['power'])
            self.dev.frequency_range(st['freq'])
            self.dev.num_points(st['npts'])
            self.dev.bandwidth(st['bw'])
            self.dev.measure(st['meas'])
            self.markers = st['markers'].copy()
            print('loaded config')
            show_state(st)

        elif 'delete'.startswith(cmd):
            for name in args:
                print('deleting config:')
                st = self.saved_states[name]
                del self.saved_states[name]
                show_state(st)

    def _cmd_trigger_source(self, *args):
        return self.dev.trigger_source(*args)

    def _cmd_npoints(self, *args):
        return self.device_command('num_points', *args)

    def _cmd_write(self, *args):
        filename = args[0]

        self._cmd_update(self.freq, self.z)
        data = {'f': self.freq, 'z': self.z}
        np.save(filename, data)

    def _cmd_read(self, *args):
        filename = args[0]
        data = np.load(filename, allow_pickle=True).item()
        self.freq, self.z = data['f'], data['z']

        self._cmd_update(self.freq, self.z)

    def _cmd_display(self, *args):
        args = [self.parse_bool(x) for x in args]
        return self.dev.display(*args)

    def _cmd_phase(self, *args):
        if not args:
            return self.phase
        
        val, rel = self.parse_number(args[0])
        if rel:
            self.phase += val
        else:
            self.phase = val
        
        return self.phase

    def status(self):
        dbm = self._cmd_power()
        span = self._cmd_span()
        np = self._cmd_npoints()

        print(f'{dbm} dBm  span: {span}  {np} points')


