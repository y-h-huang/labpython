from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxDB_Logger:
    '''Helper class for reading from and writing to InfluxDB server'''

    def __init__(self, url=None, org=None, token=None, timeout=5000):
        '''Creates a logger instance.
           Parameters url, org, and token default to the Cirrus server.'''

        if url is None:
            url = 'https://davis-cloud.physics.ualberta.ca:8086'

        if org is None:
             org='davis_group'

        if token is None:
            token='BnVmiGHjUnmr0a7XPzfrbnDcdFzPn2o100YTyieEPJqo' \
                  'kAjW4hj3OeEHD94OxbpbYvY0C2kA7uiBnWuA30_Vhw=='

        self.url, self.org, self.token, self.timeout = url, org, token, timeout
        self._client = None
        self._writer = None
        self._reader = None


    @property
    def client(self):
        '''Create a InfluxDBClient on demand.'''
        if self._client is None:
            self._client = InfluxDBClient(url=self.url,
                                          org=self.org,
                                          token=self.token, 
                                          timeout=self.timeout)
        return self._client

    def write(self, bucket, name, data, tags=None):
        '''
        Writes data to the InfluxDB server.
           bucket: name of the bucket
           name: measurement name ('_measurement' in the data explorer)
           data: a dictionary with name/value pairs
           tags: a dictionary with name/value pairs for optional tags
        '''

        if self._writer is None:
            self._writer = self.client.write_api(write_options=SYNCHRONOUS)

        pt = Point(name)

        if tags is not None:
            for k, v in tags.items():
                pt.tag(k, v)

        for k, v in data.items():
            pt.field(k, v)

        self._writer.write(bucket=bucket, record=[pt])

    def build_query(self, bucket, name, fields, tags=None, start=86400, stop=0, last_only=True):
        '''
        Returns a function that can be called to read data from the InfluxDB server.
           bucket: name of the bucket
           name: measurement name ('_measurement' in the data explorer)
           fields: a list of variable names to read
           tags: a dictionary with name/value pairs for optional tags
           start: begining of time span of interest, in seconds before now
           stop: end of time span of intetest
           last_only: read only the newest value. Still need to be in [-start, -stop] range
        '''

        if self._reader is None:
            self._reader = self.client.query_api()

        # build query string
        q = []
        q.append(f'from(bucket:"{bucket}")')
        q.append(f'range(start: {-start}s, stop: {-stop}s)')

        if last_only:
            q.append('last()')

        q.append(f'filter(fn: (r)=>r["_measurement"] == "{name}")')

        if tags is not None:
            cmds = ' or '.join(f'r["{k}"] == "{v}"' for k, v in tags.items())
            q.append('filter(fn:(r) => ' + cmds + ')')

        cmds = ' or '.join(f'r._field == "{k}"' for k in fields)
        q.append('filter(fn:(r) => ' + cmds + ')')

        qstr = '\n|>'.join(q)

        def query_func():
            output = {}

            for table in self._reader.query(org=self.org, query=qstr):
                for rec in table.records:
                    t, f, v = rec.get_time(), rec.get_field(), rec.get_value()

                    if not f in output:
                        output[f] = {'time': [], 'value': []}

                    output[f]['time'].append(t)
                    output[f]['value'].append(v)

            return output

        return query_func


    def read(self, bucket, name, fields, tags=None, start=86400, stop=0, last_only=True):
        ''' Builds a query and read once; calling semantics same as in build_query()'''
        return self.build_query(bucket, name, fields, tags, start, stop, last_only)()

def test():
    import time, sys
    import math
    from itertools import count

    logger = InfluxDB_Logger()
    bucket = 'playground'
    meas = 'RandomData'

    def write_test():
        for i in count():
            x = math.sin(i/30)
            y = math.cos(i/53 + 1.1)*2
            logger.write(bucket=bucket,
                         name=meas,
                         tags={'Stage 2': 'Stuff'},
                         fields={'x': x, 'y': y})
            time.sleep(1)

    def read_test():
        res = logger.read(bucket=bucket,
                          name=meas,
                          tags={'Stage 2': 'Stuff'},
                          fields='x y'.split(),
                          last_only=False)
        print(res)

    def query_test():
        func = logger.build_query(bucket=bucket,
                          name=meas,
                          tags={'Stage 2': 'Stuff'},
                          fields='x y'.split(),
                          last_only=True)

        for _ in range(10):
            res = func()
            print('x:', res['x']['value'])
            print('y:', res['y']['value'])
            print()
            time.sleep(1)

    test_ok = True
    if len(sys.argv) <= 1:
        test_ok = False
    elif sys.argv[1] == 'write':
        write_test()
    elif sys.argv[1] == 'read':
        read_test()
    elif sys.argv[1] == 'query':
        query_test()

    if not test_ok:
        print('To perform test, pass "read", "write", or "query" as commandline argument')

if __name__ == '__main__':
    test()
