from . import HF2LI

if __name__ == '__main__':

    settings = {
        'sigins': {
            '0': {
                'ac': 0,
                'range': 1,
            },
            '1': {
                'ac': 0,
                'range': 0.1,
            },
        },
        'demods': {
            '0': {
                'rate': 1e6,
                'enable': 1,
                'adcselect' : 0,
                'order': 4,
                'timeconstant': 1e-5,
                'oscselect': 0,
                'harmonic': 1,
            },
            '1': {
                'rate': 1e6,
                'enable': 1,
                'adcselect' : 1,
                'order': 4,
                'timeconstant': 1e-5,
                'oscselect': 0,
                'harmonic': 1,
            },
        },
        'oscs': {
            '0': {
                'freq': 12.756e6,
            },
        },
        # 'scopes/0': {
        #     'enable': 1,
        #     'channel': 1,
        #     'time': 10,
        #     'bwlimit': 1,
        # },
    }

    dev = HF2LI('dev538', settings)
    #print(dev.sigouts[0].enables[0])
    #dev.sigouts[0].enables[1] = 0
    #dev.sigouts[1].enables[1] = 1
    #print(dev.clockbase)a

    scope = dev.scope()
    scope.set({
            'averager': {
                'weight': 1,
                'restart': 0
            },
        })

    print(scope.read(channel=0, error_redo=False))

    spectrum = dev.spectrum()
    print(spectrum.grid_cols)
