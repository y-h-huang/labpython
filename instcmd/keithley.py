from instcmd.instr_cmd import DeviceCommandLine
import numpy as np

class cmd_KEITHLEY(DeviceCommandLine):
    devname = 'cmd_KEYTHLEY'

    def _cmd_output(self, on=None):
        if on is not None:
            on = self.parse_bool(on)
            self.dev.output(on)
        return 'On' if self.dev.output() else 'Off'


    def _cmd_on(self):
        return self._cmd_output('On')

    def _cmd_off(self):
        return self._cmd_output('Off')

    def _cmd_set_current(self, *args):
        return self.device_command('current', *args)
 

    def _cmd_degauss(self, *args):
        if not args:
            I = self.dev.current()
        else:
            val, rel = self.parse_number(args[0])

            if rel:
                I = self.dev.current() + val
            else:
                I = val
            
        import time

        period = 0.2
        omega = 2*np.pi/period
        decay = 5.0
        swing = 1.0

        t0 = time.time()

        while True:
            t = time.time() - t0

            #amp = swing*np.exp(-t/decay)
            amp = swing*(1 - t/decay)
            dI = np.cos(omega*t)*amp
            if amp < 1e-2:
                break

            self.dev.current(I + dI)
            time.sleep(0.001)

        self.dev.current(I)

    def _cmd_measure_current(self, *args):
        return self.device_command('measure', *args)
    

    def status(self):
        output = self._cmd_output()
        Iset = self._cmd_set_current()
        Imeas = self._cmd_measure_current()

        return f'{output}   set {Iset} mA   actual {Imeas:.5f} mA'
