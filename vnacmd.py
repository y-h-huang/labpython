from instcmd.BNC845 import cmd_BNC
from instruments.rfsource import BNC845
from instruments.Keithley2400 import KEITHLEY
from instruments.vna import VNA
from instruments.instrument_set import InstrumentSet
from instruments.devicelist import device_ids
from instcmd import InstrumentCommands

#('vna', VNA, 'USB0::0x2A8D::0x5D01::MY54505262::0::INSTR')
#('rf', BNC845, 'USB0::0x03EB::0xAFFF::4C1-3A3200905-1225::0::INSTR'),
iset = InstrumentSet()
for addr, idn in device_ids():
    if idn is None:
        continue

    #print(f'Device {idn} at address {addr}')
    if idn.find('Keysight') != -1 and idn.find('E5063A') != -1:
        iset.add(name='vna', cls=VNA, addr=addr)
    elif idn.find('Berkeley Nucleonics') != -1:
        iset.add(name='rf', cls=BNC845, addr=addr)
    elif idn.find('KEITHLEY') != -1 and idn.find('MODEL 2400') != -1:
        iset.add(name='psu', cls=KEITHLEY, addr=addr)

with iset.run() as devs:
    cmd = InstrumentCommands().add_set(devs)

    for res in cmd.run():
        print(res)
        ctrl_c = False
