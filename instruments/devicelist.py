import pyvisa
import time

def device_ids():
    rm = pyvisa.ResourceManager()

    devs = []
    addrs = []
    ids = []

    for addr in rm.list_resources():
        try:
            dev = rm.open_resource(addr)
            devs.append((addr, dev))
            dev.write('*IDN?')
        except Exception:
            #print(f'Failed to open device at address {addr}')
            continue

    for addr, dev in devs:
        try:
            idn = dev.read().rstrip()
        except Exception:
            idn = None

        yield addr, idn
        dev.clear()
        dev.close()

    rm.close()

if __name__ == '__main__':
    for addr, idn in device_ids():
        print(addr)
        print(idn)
        print()