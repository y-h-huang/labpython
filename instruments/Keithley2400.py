# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:33:21 2020

@author: cpott_000
"""


import numpy as np
import time
import pyvisa as visa

class KEITHLEY:

    DEVICE_NAME = 'KEITHLEY2400'

    def __init__(self, rm, address):
        self.address = address
        self.rm = rm
        self.dev = rm.open_resource(address)
        self.dev.clear()
        print(self.dev.query('*IDN?', delay=1).rstrip())
        self.dev.timeout=5*1000
    
    def current_limit(self, *args):
        if len(args) == 0:
            return self.limits
        
        if len(args) == 1:
            I = abs(args[0])
            self.limits = [-I, I]
        else:
            self.limits = args[:2]

    def current(self, I=None, Imax=100):
        if I is None:
            return float(self.dev.query(':SOUR:CURR:LEV?'))*1000

        Imax = abs(Imax)
        I = min(I, Imax)
        I = max(I, -Imax)

        self.dev.write(f':SOUR:CURR:LEV {I/1000:.6f}')
    
    def measure(self, mode='I'):
        cmd = ':MEAS:CURR:DC?'
        res = self.dev.query(cmd, delay=1)
        return float(res.split(',')[1])*1000
    

    def output(self, state=None):
        if state is None:
            return self.dev.query(':OUTP:STAT?').strip() == '1'

        self.dev.write(':OUTP:STAT ' + ('ON' if state else 'OFF'))


    def close(self):
        self.dev.close()


if __name__ == '__main__':
    rm = visa.ResourceManager()
    psu = KEITHLEY(rm)
    psu.output(True)
    # psu.current(0.006)
    try:
        print(psu.meas())
    finally:
        psu.close()
        rm.close()
        
        
