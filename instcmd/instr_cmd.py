import inspect

class DeviceCommandLine():
    devname = 'Nonce Device'


    def __init__(self, dev, name):
        self.dev = dev
        self.name = name
        self.cmds = {}

        # cololect command names
        for n, f in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if not n.startswith('_cmd_'):
                continue
            n = n[5:]
            self.cmds[n] = f


    def find_command(self, cmd=None):
        results = []

        cmd = cmd.lower()
        for n, f in self.cmds.items():
            if n == cmd:
                return f

            if n.startswith(cmd):
                results.append((n, f))

        if len(results) == 0:
            raise AttributeError(f'No such command: `{cmd}\'')

        if len(results) > 1:
            raise AttributeError(f'Ambiguous command `{cmd}\', candidates are\n\t'
                                + '\n\t'.join(n for n, _ in results))

        return results[0][1]


    def command(self, cmd=None, *args):

        if cmd is None:
            return self.status()

        if cmd == 'cmds':
            for n in sorted(self.cmds.keys()):
                print(n)
            return

        try:
            return self.find_command(cmd)(self, *args)
        except AttributeError as e:
            print('Attribute error:', e)
            raise
        except Exception as e:
            print(f'Command {cmd} {args} failed')
            print(e)
            #raise


    def device_command(self, cmd, argstr=None):
        func = getattr(self.dev, cmd)
        if argstr is None:
            return func()

        val, rel = self.parse_number(argstr)

        if rel:
            current_value = func()
            val += current_value
        
        func(val)
        return func()


    def parse_number(self, s):
        relative = 0
        mul = 1

        if s.endswith('+'):
            relative = 1
            s = s[:-1]
        elif s.endswith('-'):
            relative = -1
            s = s[:-1]

        if s.endswith('G'):
            mul = 1e9
        elif s.endswith('M'):
            mul = 1e6
        elif s.endswith('k') or s.endswith('K'):
            mul = 1e3
        elif s.endswith('m'):
            mul = 1e-3
        elif s.endswith('u'):
            mul = 1e-6
        elif s.endswith('n'):
            mul = 1e-9
        
        if mul != 1:
            s = s[:-1]
        
        if relative:
            mul *= relative

        return float(s)*mul, relative


    def parse_bool(self, s):
        s = s.lower()
        if s == '0' or s == 'off' or s == 'false':
            return False
        if s == '1' or s == 'on' or s == 'true':
            return True

        raise ValueError(f'Don\'t know how to convert `{s}\' to boolean')
