import pyvisa

class DC205:
    
    DEVICE_NAME = 'DC205'

    def __init__(self, rm, address):

        self.dev = rm.open_resource(address)
        self.dev.baud_rate = 115200
        self.dev.clear()
        print(self.query('*IDN?'))

    def query(self, cmd):
        return self.dev.query(cmd).strip()

    def write(self, cmd):
        self.dev.write(cmd)
        return self

    def close(self):
        self.dev.close()

    @property
    def interlock(self):
        return self.query('ILOC?') == '1'

    @property
    def overload(self):
        return self.query('OVLD?') == '1'

    def status(self, bit=None):
        cmd = '*STB?' if bit is None else f'*STB?{bit}'
        return self.query(cmd)

    def range(self, volt=None):
        ranges = [1, 10, 100]
        if volt is None:
            return ranges[int(self.query('RNGE?'))]

        self.write(f'RNGE {ranges.index(volt)}')
        return self

    def floating(self, flt=None):
        if flt is None:
            return self.query('ISOL?') == '1'

        flt = '1' if flt else '0'
        self.write(f'ISOL {flt}')
        return self

    def output(self, onoff=None):
        if onoff is None:
            return self.query('SOUT?') == '1'

        onoff = '1' if onoff else '0'
        self.write(f'SOUT {onoff}')
        return self

    def voltage(self, v=None):
        if v is None:
            return float(self.query('VOLT?'))

        self.write(f'VOLT {v}')
        return self

if __name__ == '__main__':
    rm = pyvisa.ResourceManager()
    dc = DC205(rm, 'COM4')

    import time
    import numpy as np

    dc.output(False)
    
    for v in np.linspace(0, 10, 21):
        dc.voltage(v)
        dc.floating(False)
        dc.output(True)
        time.sleep(1)
        dc.floating(True)
        dc.output(False)
        time.sleep(.5)

    while True:
        cmd = input('> ')
        if cmd == 'exit':
            break

        try:
            res = dc.query(cmd)
            print(res)
        except pyvisa.errors.VisaIOError:
            print('IOError')

    dc.close()
    rm.close()
