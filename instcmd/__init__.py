class InstrumentCommands():
    def __init__(self, *devs):
        self.devices = {}
        for n, d in devs:
            self.add(n, d)
        
        self.auto_cmds = []
        self.auto_cmd_active = True

        self.result = None


    def do_auto_commands(self, *args):
        if not args:
            print(f'Auto commands: {"On" if self.auto_cmd_active else "Off"}')
            for i, a in enumerate(self.auto_cmds):
                print(f'{i:3d}: {" ".join(a)}')
            return

        if 1:
            cmd = args[0].lower()
            args = args[1:]

            if 'add'.startswith(cmd):
                try:
                    idx = int(args[0])
                    args = args[1:]
                except Exception:
                    idx = len(self.auto_cmds)

                self.auto_cmds.insert(idx, tuple(args))

            elif 'delete'.startswith(cmd):
                for idx in args:
                    if idx == '*':
                        self.auto_cmds = []
                        return
                    else:
                        print(f'delete {int(idx)}')
                        self.auto_cmds.pop(int(idx))

            elif 'replace'.startswith(cmd):
                idx = int(args[0])
                self.auto_cmds[idx] = tuple(args[1:])

            elif cmd == 'off':
                self.auto_cmd_active = False

            elif cmd == 'on':
                self.auto_cmd_active = True

            else:
                print(f'Bad argument to auto: `{cmd}\'')


    @staticmethod
    def load_command_interface(dev):
        s = dev.__class__.DEVICE_NAME

        if s == 'BNC845':
            from .BNC845 import cmd_BNC
            return cmd_BNC(dev, s)
        
        if s == 'KEITHLEY2400':
            from .keithley import cmd_KEITHLEY
            return cmd_KEITHLEY(dev, s)

        if s == 'VNA':
            from .vna import cmd_VNA
            return cmd_VNA(dev, s)
        
        raise ValueError(f'Device {dev} has no command line interface')


    def add_set(self, iset):
        for n, d in zip(iset._fields, iset):
            self.add(n, d)
        return self


    def add(self, name, dev):
        self.devices[name] = self.load_command_interface(dev)
        self.devices[name].inst_cmds = self

        return self


    def list_devices(self):
        return list(self.devices.keys())


    def auto_exec(self):
        prev_result = None

        for dn, *args in self.auto_cmds:
            args = [x if x != '#' else prev_result for x in args]

            res = self.devices[dn].command(*args)
            if res is not None:
                prev_result = str(res)

        return prev_result


    def run(self):
        while True:
            try:
                text = input('> ')
                res = None
                text = text.rstrip()

                for s in text.split(';'):
                    if not s: continue
                    devname, *args = s.strip().split()

                    if devname == 'exit' or devname == 'quit':
                        return

                    cmd_res = self.run_single(devname, *args)

                if not text.endswith(';') and self.auto_cmd_active:
                    prev_result = self.auto_exec()
                else:
                    prev_result = cmd_res


                if cmd_res is not None:
                    yield cmd_res

            except Exception as e:
                print(e)
            except KeyboardInterrupt:
                print()
                pass


    def load_module(self, dev_name, mod_name):
        import importlib
        print(f'importing {mod_name} as {dev_name}')
        mod = importlib.import_module(mod_name)
        self.devices[dev_name] = mod.load(self, dev_name)


    def run_single(self, devname, *args):
        if not devname:
            return

        match devname:
            case 'auto':
                self.do_auto_commands(*args)
                return

            case 'eval':
                res = eval(' '.join(args))
                self.result = str(res)
                return res

            case 'module':
                self.load_module(*args)
                return

        args = [x if x != '#' else self.result for x in args]

        if devname not in self.devices:
            raise ValueError(f'Unknown device name `{devname}\'')

        res = self.devices[devname].command(*args)
        if res is not None:
            self.result = str(res)
        
        return res
