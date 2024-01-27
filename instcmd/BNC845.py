from instcmd.instr_cmd import DeviceCommandLine

class cmd_BNC(DeviceCommandLine):
    devname = 'cmd_BNC845'

    def _cmd_output(self, on=None):
        if on is not None:
            on = self.parse_bool(on)
            self.dev.output(on)
        return 'On' if self.dev.output() else 'Off'


    def _cmd_power(self, *args):
        return self.device_command('power', *args)
    

    def _cmd_frequency(self, *args):
        return self.device_command('frequency', *args)


    def _cmd_phase(self, *args):
        return self.device_command('phase', *args)

    def _cmd_rad(self, *args):
        return self.device_command('rad', *args)

    def status(self):
        output = self._cmd_output()
        dbm = self._cmd_power()
        freq = self._cmd_frequency()

        return f'{output}   {dbm} dBm   {freq} Hz'

    def _cmd_on(self):
        return self._cmd_output('On')
    
    def _cmd_off(self):
        return self._cmd_output('Off')

    def _cmd_lock(self, mode=None, freq=None):
        if mode is None:
            return self.dev.lock()

        if freq is not None:
            freq = int(freq)

        print(f'lock external={self.parse_bool(mode)} freq={freq}')
        return self.dev.lock(self.parse_bool(mode), freq=freq)

    def _cmd_refout(self, out=None):
        if out is None:
            return self.dev.ref_out()
        else:
            out = int(out)

        self.dev.ref_out(out)
