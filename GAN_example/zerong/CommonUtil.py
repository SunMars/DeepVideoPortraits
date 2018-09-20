from __future__ import absolute_import, division, print_function
import datetime


class Logger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename):
        assert self.file is None
        self.file = open(filename, 'wt')
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, msg):
        now = datetime.datetime.now()
        dtstr = now.strftime('%Y-%m-%d %H:%M:%S')
        t_msg = '[%s] %s' % (dtstr, msg)

        print(t_msg)
        if self.file is not None:
            self.file.write(t_msg + '\n')
        else:
            self.buffer += t_msg

    def flush(self):
        if self.file is not None:
            self.file.flush()

logger = Logger()