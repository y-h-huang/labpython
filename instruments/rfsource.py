# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:02:29 2020

Class to control the BNC model 845 RF source.

@author: Emil
"""

import time
import pyvisa
from math import pi

class BNC845:
    """Very basic interface to the BNC Model 845 RF source."""

    DEVICE_NAME = 'BNC845'

    def __init__(self, rm, address='USB0::0x03EB::0xAFFF::471-4396D0000-1219::0::INSTR'):
        self.rm = rm
        self.address = address
        self.dev = rm.open_resource(address)
        self.dev.clear()

        for _ in range(5):
            try:
                time.sleep(.1)
                idn = self.dev.query('*IDN?').rstrip()
                print(idn)
                break
            except pyvisa.errors.VisaIOError:
                print('BNC845: error on IDN check, retrying')
                pass
        else:
            print('Failed to identify rfsource')
            raise

        self.auto_off = True


    def close(self):
        """Close the VISA session."""
        if self.auto_off:
            self.output(False)
        self.dev.close()
    

    def frequency(self, set_freq=None):
        """Set or query the frequency in Hz."""
        if set_freq is None:
            return float(self.dev.query(':FREQ?'))
        self.dev.write(':FREQ {:.0f}'.format(set_freq))


    def phase(self, phase=None):
        ''' phase in degrees '''
        if phase is None:
            return self.rad()/pi*180

        self.rad(phase*pi/180)

    def rad(self, phase=None):
        ''' phase in degrees '''
        if phase is None:
            return float(self.dev.query(':PHAS?'))

        self.dev.write(f':PHAS {phase}')

    def output(self, set_status=None):
        """Set or query the output status."""
        if set_status is None:
            stat = int(self.dev.query(':OUTP?'))
            return bool(stat)
        if set_status:
            self.dev.write(':OUTP ON')
        else:
            self.dev.write(':OUTP OFF')
    

    def power(self, set_power=None):
        """Set or query the output power in dBm"""
        if set_power is None:
            return float(self.dev.query(':POW:LEV?'))
        self.dev.write(':POW:LEV {:.3f}'.format(set_power))


    def lock(self, ext_lock=None, freq=10_000_000):
        if ext_lock is None:
            src = self.dev.query(':ROSC:SOUR?')[:-1]
            if src == 'INT':
                return False, None

            freq = int(float(self.dev.query(':ROSC:EXT:FREQ?').rstrip()))
            locked = self.dev.query(':ROSC:LOCK?').rstrip() != '0'
            return locked, freq

        if ext_lock:
            self.dev.write(':ROSC:SOUR EXT')
            self.dev.write(':ROSC:EXT:FREQ {freq:d}')
            self.dev.write(':ROSC:LOCK:TEST')
            time.sleep(5)

            freq = int(float(self.dev.query(':ROSC:EXT:FREQ?').rstrip()))
            locked = self.dev.query(':ROSC:LOCK?').rstrip() != '0'
            return locked, freq

        else:
            self.dev.write(':ROSC:SOUR INT')


    def ref_out(self, out_freq=None):
        if out_freq is None:
            stat = self.dev.query('ROSC:OUTP?').rstrip() != '0'
            freq = int(float(self.dev.query('ROSC:OUTP:FREQ?')))
            return stat, freq
        if not out_freq:
            self.dev.write('ROSC:OUTP 0')
        else:
            self.dev.write('ROSC:OUTP 1')
            self.dev.write(f'ROSC:OUTP:FREQ {out_freq:d}')


if __name__ == '__main__':
    import pyvisa
    rm = pyvisa.ResourceManager()
    rf = BNC845(rm, 'USB0::0x03EB::0xAFFF::471-4396D0600-1250::0::INSTR')
    # rf.frequency(2.78e9)
    # rf.power(16)
    # rf.output(True)
    print(rf.frequency())
    print(rf.output())
    print(rf.power())
    rf.close()
