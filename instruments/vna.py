# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:58:45 2020

@author: ev
"""

import pyvisa as visa
import numpy as np
import time
import sys

class VNA:
    """Some basic VNA functions. Most of the code stolen from Vaisakh's/Robyn's code."""

    DEVICE_NAME = 'VNA'

    def __init__(self, visa_rm, address="USB0::0x2A8D::0x5D01::MY54505262::0::INSTR"):
        self.address = address
        self.rm = visa_rm
        self.vna = visa_rm.open_resource(address)

        self.vna.clear()
        self.off()
        
        for _ in range(3):
            time.sleep(1)
            try:
                print(self.vna.query('*IDN?').rstrip())
            except visa.errors.VisaIOError:
                print('Error identifying VNA, retrying')
            else:
                break
        else:
            raise

        self.vna.timeout=20*60*1000
    
    def close(self):
        self.off()
        self.vna.write(":TRIG:SOUR INT")
        self.vna.clear()
        self.vna.close()

    def off(self):
        self.vna.write(":INIT1:CONT OFF")
        

    def on(self):
        self.vna.write(":INIT1:CONT ON")


    def measure(self, S=None):
        if S is None:
            return self.vna.query('CALC1:PAR1:DEF?').rstrip()
        
        self.vna.write(f':CALC1:PAR1:DEF {S}')
        self.vna.write(f':CALC1:PAR2:DEF {S}')
        self.vna.write(f':CALC1:PAR3:DEF {S}')


    def display(self, on=None):
        if on is None:
            return self.vna.query(':DISP:ENAB?').startswith('1')
        
        self.vna.write(':DISP:ENAB ' + ('ON' if on else 'OFF'))
        if on:
            self.vna.write('DISP:UPD')


    def setup(self, S=None): # S='S11' etc
        # set number of traces
        self.vna.write(":CALC1:PAR:COUN 3")

        # window display setting
        self.vna.write(":DISP:WIND1:SPL D13_23")

        if S is not None:
            # set S parameter for each trace
            self.vna.write(":CALC1:PAR1:DEF {}".format(S))
            self.vna.write(":CALC1:PAR2:DEF {}".format(S))
            self.vna.write(":CALC1:PAR3:DEF {}".format(S))

        # Choose data format
        self.vna.write(":CALC1:PAR1:SEL")
        self.vna.write(":CALC1:FORM MLOG")

        self.vna.write(":CALC1:PAR2:SEL")
        #self.vna.write(":CALC1:FORM PHAS")
        self.vna.write(":CALC1:FORM UPH")

        self.vna.write(":CALC1:PAR3:SEL")
        self.vna.write(":CALC1:FORM POL")

        # turn averaging off
        #self.vna.write(":SENS1:AVER OFF")

    def output(self, onoff=None):
        if onoff is not None:
            s = 'ON' if onoff else 'OFF'
            self.vna.write(f':OUTP {s}')

        res = self.vna.query(':OUTP?').strip()
        return res == '1'


    def phase_offset(self, p=None):
        cmd = ':CALC1:CORR:OFFS:PHAS'
        if p is None:
            return float(self.vna.query(cmd + '?'))
        
        self.vna.write(f'{cmd} {p}')

    def power(self, set_power=None):
        """Set or query the output power in dBm"""
        if set_power is None:
            return float(self.vna.query(':SOUR1:POW:LEV?'))

        self.vna.write(':SOUR1:POW:LEV {:.3f}'.format(set_power))


    def trigger_source(self, trig=None):
        if trig is not None:

            t = trig.lower()
            if t == 'int':
                self.vna.write(':TRIG:SOUR INT')
            elif t == 'BUS':
                self.vna.write(':TRIG:SOUR BUS')
            else:
                raise ValueError(f'Unknown trigger source type: {trig}')
        
        return self.query(':TRIG:SOUR?').rstrip()

    def sweep_current_setting(self):
        self.setup()

        #set trigger to cts
        self.vna.write(':INIT1:CONT ON')
        self.vna.write(':TRIG:SOUR BUS')
        self.vna.write(':TRIG:SING')

        self.vna.query('*OPC?')

        #Autoscale
        #self.vna.write(':DISP:WIND1:TRAC1:Y:AUTO')
        #self.vna.write(':DISP:WIND1:TRAC2:Y:AUTO')
        #self.vna.write(':DISP:WIND1:TRAC3:Y:AUTO')

        self.vna.write(':FORM:BORD NORM')
        self.vna.write(':FORM:DATA REAL')

        data = self.vna.query_binary_values(':CALC1:DATA:FDAT?', datatype='d', is_big_endian=True)
        freq = np.array(self.vna.query_binary_values(':SENS1:FREQ:DATA?', 'd', is_big_endian=True))

        self.vna.write(':FORM:DATA ASC')

        z = np.array(data).view(dtype=np.complex128)
        return freq, z

    def autoscale(self):
        self.vna.write(':DISP:WIND1:TRAC1:Y:AUTO')
        self.vna.write(':DISP:WIND1:TRAC2:Y:AUTO')
        self.vna.write(':DISP:WIND1:TRAC3:Y:AUTO')

    def sweep(self, start, stop, num_points=10001, bw=10e3, only_z=False):
        self.setup()

        self.vna.write(f':SENS1:FREQ:STAR {start}')
        self.vna.write(f':SENS1:FREQ:STOP {stop}')
        self.vna.write(f':SENS1:SWE:POIN {num_points}')
        self.vna.write(f':SENS1:BWID {bw}')
        
        #set trigger to cts
        self.vna.write(':INIT1:CONT ON')
        self.vna.write(':TRIG:SOUR BUS')
        self.vna.write(':TRIG:SING')

        self.vna.query('*OPC?')

        #self.autoscale()

        self.vna.write(':FORM:BORD NORM')
        self.vna.write(':FORM:DATA REAL')

        data = self.vna.query_binary_values(':CALC1:DATA:FDAT?', datatype='d', is_big_endian=True)

        if not only_z:
            freq = np.array(self.vna.query_binary_values(':SENS1:FREQ:DATA?', 'd', is_big_endian=True))

        self.vna.write(':FORM:DATA ASC')

        z = np.array(data).view(dtype=np.complex128)
        if only_z:
            return z
        return freq, z


    def sweep_segments(self, segments):
        # stim is alwasy start/stop, ifbw is always specified, power/delay/time are always default
        cmds = "5 0 1 0 0 0".split()
        cmds.append(f'{len(segments)}')
        for s in segments:
            for k in 'start stop num_points bw'.split():
                cmds.append(str(s[k]))

        self.vna.write(':SENS1:SWE:TYPE SEGM')
        self.vna.write(':DISP:WIND1:X:SPAC LIN')

        self.vna.write(':SENS1:SEGM:DATA ' + ','.join(cmds))
        self.vna.write(':INIT1:CONT ON')
        self.vna.write(':TRIG:SOUR BUS')
        self.vna.write(':TRIG:SING')

        self.vna.query('*OPC?')

        #Autoscale
        self.vna.write(':DISP:WIND1:TRAC1:Y:AUTO')
        self.vna.write(':DISP:WIND1:TRAC2:Y:AUTO')
        self.vna.write(':DISP:WIND1:TRAC3:Y:AUTO')

        self.vna.write(':FORM:BORD NORM')
        self.vna.write(':FORM:DATA REAL')

        freq = np.array(self.vna.query_binary_values(':SENS1:FREQ:DATA?', 'd', is_big_endian=True))
        data = self.vna.query_binary_values(':CALC1:DATA:FDAT?', datatype='d', is_big_endian=True)

        self.vna.write(':SENS1:SWE:TYPE LIN')
        self.vna.write(':FORM:DATA ASC')

        return freq, np.array(data).view(dtype=np.complex128)

    def combo_sweep(self, f0, f1, n_pts, f_noisy_lo, f_noisy_hi, normal_bw, noisy_bw):

        f = np.linspace(f0, f1, n_pts)

        below = f < f_noisy_lo
        above = f > f_noisy_hi
        between = ~(below | above)

        idices = [(idx, bw) for idx, bw in zip((below, between, above), (normal_bw, noisy_bw, normal_bw)) if np.sum(idx) > 0]
        segs = []
        for idx, bw in idices:
            fidx = f[idx]
            segs.append({'start': fidx[0], 'stop': fidx[-1], 'num_points': len(fidx), 'bw': bw})

        return self.sweep_segments(segs)

    def sweep_cs(self, center, span, num_points=10001, bw=10e3):
        return self.sweep(center-span/2, center+span/2, num_points, bw)

    def frequency_start(self, freq=None):
        cmd = ':SENS1:FREQ:START'
        if freq is None:
            return float(self.vna.query(cmd + '?').rstrip())

        self.vna.write(cmd + f' {freq:.10e}')

    def frequency_stop(self, freq=None):
        cmd = ':SENS1:FREQ:STOP'
        if freq is None:
            return float(self.vna.query(cmd + '?').rstrip())

        self.vna.write(cmd + f' {freq:.10e}')

    def frequency_range(self, r=None):
        if r is None:
            return self.frequency_start(), self.frequency_stop()

        self.frequency_start(r[0])
        self.frequency_stop(r[1])

    def trigger_source(self, src=None, mode=None):
        if src is None:
            return self.vna.query(':TRIG:SOUR?').rstrip()
        self.vna.write(f':TRIG:SOUR {src}')

    # get/set bandwidth setting (under AVG menu)
    def bandwidth(self, value=None):
        cmd = f':SENS1:BAND'
        if value is None:
            response = self.vna.query(cmd + '?')
            return int(float(response.rstrip()))
        else:
            valid = [int(x) for x in '10|15|20|30|40|50|70|100|150|200|300|400|500|700|1k|1500|2k|3k|4k|5k|7k|10k|15k'
                                     '|20k|30k|40k|50k|70k|100k|150k|200k|300k'.replace('k', '000').split('|')]
            nearest = min([(abs(x - value), x) for x in valid])[1]
            self.vna.write(cmd + f' {nearest}')

            v = self.bandwidth()
            if v != value:
                sys.stderr.write(f'[INFO] Bandwidth value "{value}" coerced into {v}')

    def num_points(self, n=None):
        if n is None:
            return int(self.vna.query(':SENS1:SWE:POIN?').rstrip())

        self.vna.write(f':SENS1:SWE:POIN {int(n):d}')

    def electrical_delay(self, trace=None, delay=None):
        if trace is None:
            cmd = ':CALC1:CORR:EDEL:TIME'
        else:
            cmd = f':CALC1:TRAC{trace}:CORR:EDEL:TIME'

        if delay is None:
            return float(self.vna.query(cmd + '?').rstrip())

        self.vna.write(f'{cmd} {delay}')

if __name__ == '__main__':
    rm = visa.ResourceManager('C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\agvisa\\agbin\\visa32.dll')
    vna = VNA(rm)
    data = vna.sweep(1e9, 3e9)
    
