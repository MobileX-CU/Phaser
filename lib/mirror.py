import collections
import pandas as pd
import serial
import time
import logging

class NativeMirror(object):
    def __init__(self):
        raise NotImplementedError('Mirror control via native SDK is not yet implemented')
    # import optoMDC
    # import optoKummenberg
    # mre2 = optoMDC.connectmre2(port='/dev/cu.usbmodem00000000001A1', boot_in_simple=False, board_reset=True)
    # mre2.SerialMirror.GetConnectedStatus()
    # chs = [mre2.SerialMirror.Channel_0, mre2.SerialMirror.Channel_1]
    # for ch in chs:
    #     ch.StaticInput.SetAsInput()
    #     ch.SetControlMode(optoMDC.Units.XY)
    # xy_scale = 0.5
    # xy = (-1*xy_scale, -0.075*xy_scale)
    # for i, ch in enumerate(chs):
    #     si = ch.StaticInput
    #     si.SetXY(xy[i])
    #     print(si.GetXY())
    # mre2.disconnect()

class SerialMirror(object):

    # Mirror inventory
    Mirrorcle = collections.namedtuple('Mirrorcle', 'pn halfmech bias diff bw')
    Optotune = collections.namedtuple('Optotune', 'pn halfmech')
    Models = dict(
        AU_3600_T20 = Mirrorcle('A7B1.1-3600AU-TINY20.4-A/W/EP', 6.6, 80, 135, 120),
        AL_2400_T48 = Mirrorcle('A5M24.3-2400AL-TINY48.4-A/W/EP', 7.0, 90, 150, 300),
        AU_4600_T48 = Mirrorcle('A5L3.3(C2)-4600AU-TINY48.4-A/W/EP', 5.5, 90, 178, 160),
        AU_6400_T48 = Mirrorcle('A5L3.3(C1)-6400AU-TINY48.4-A/W/EP', 3.5, 90, 178, 120),
        AL_3600_T48 = Mirrorcle('A7B2.3-3600AL-TINY48.4-A/F/EP', 6.5, 80, 150, 200),
        AL_5000_T48 = Mirrorcle('A8L2.2-5000AL-TINY48.4-A/F/EP', 5.3, 90, 178, 120),
        AL_2000_T48 = Mirrorcle('A5L2.2-2000AL-TINY48.4-A/W/EP', 1.75, 90, 172, 1800),
        AU_15000_MRE2 = Optotune('MR-E-2', 25))

    def __init__(self, sn, loc, port, range_scale=1.0, mirrors_csv_path='mirrors.csv'):
        self.logger = logging.getLogger()
        
        # Initialize mirror from inventory, assumes 'mirrors.csv' is in the calling directory
        mirrors = pd.read_csv(mirrors_csv_path, index_col=0)
        if sn not in mirrors.index:
            raise ValueError('Unsupported serial number')
        mirror = mirrors.loc[sn]
        if mirror.broken:
            raise ValueError('Mirror is broken')
        elif mirror.location and mirror.location != loc:
            raise ValueError(f'Mirror already used in "{mirror.location}" != "{loc}"')
        self.model = SerialMirror.Models[mirror.model]

        # Initialize model-specific params
        range_scale = min(1.0, range_scale)
        range_scale = max(0, range_scale)
        self.range_scale = range_scale
        if type(self.model) is SerialMirror.Mirrorcle:
            self.min_delay_s = 0.001
            self.baud = 460800
            self.endl = '\n'
        elif type(self.model) is SerialMirror.Optotune:
            self.min_delay_s = 0.001
            self.baud = 256000
            self.endl = '\r\n'
        else:
            raise NotImplementedError('Model not implemented')
        
        # Create serial connection
        self.enabled = False
        self.conn = serial.Serial(
            port, 
            baudrate=self.baud, 
            bytesize=serial.EIGHTBITS, 
            parity=serial.PARITY_NONE,
            xonxoff=False, 
            rtscts=False, 
            stopbits=serial.STOPBITS_ONE, 
            timeout=1, 
            dsrdtr=True)
        
    def _send(self, cmd, pause=0, expect=None, throw_on_wrong=False):
        cmd = (cmd + self.endl).encode('utf-8')
        self.logger.debug(f'> {cmd}')
        self.conn.write(cmd)
        time.sleep(pause + self.min_delay_s)
        if expect:
            rsp = self.conn.readline().decode().strip()
            self.logger.debug(f'< {rsp} {"==" if rsp == expect else "!="} {expect}')
            if throw_on_wrong and rsp != expect:
                raise ValueError(f'Got "{rsp}", expected "{expect}"')

    def steer(self, xy, pause=0, expect=None, throw_on_wrong=False):
        if not self.enabled: raise Exception('Mirror not enabled')
        # Limit steering commands to [-1, 1]
        x = xy[0]
        x = max(-1, x)
        x = min(1, x)
        y = xy[1]
        y = max(-1, y)
        y = min(1, y)
        self.logger.info(f'Steering to {x},{y} (converted from {xy})')

        # Send command
        if type(self.model) is SerialMirror.Mirrorcle:
            self._send(f'MTI+GT {x:.4f} {y:.4f} 0', pause, expect, throw_on_wrong)
        elif type(self.model) is SerialMirror.Optotune:
            self._send(f'xy= {x*self.range_scale:.4f};{y*self.range_scale:.4f}', pause, expect, throw_on_wrong)
            

    def enable(self):
        if self.enabled: return
        self.logger.info('Enabling mirror')
        if type(self.model) is SerialMirror.Mirrorcle:
                
            # Reset any previous connections
            self.conn.reset_output_buffer()
            self.conn.reset_input_buffer()
            for _ in range(5):
                self._send('MTI+EX', pause=0.2)
            self.conn.reset_output_buffer()
            self.conn.reset_input_buffer()

            # Send setup commands
            self._send(f'$MTI$', pause=0.5, expect='MTI-Device MTI-MZ-2.3.113:USB Ready in Command Mode', throw_on_wrong=False)
            self._send(f'MTI+VB {self.model.bias:.0f}', pause=0.5, expect='MTI-OK', throw_on_wrong=True)
            self._send(f'MTI+VD {self.model.diff * self.range_scale:.0f}', pause=0.5, expect='MTI-OK', throw_on_wrong=True)
            self._send(f'MTI+BW {self.model.bw:.0f}', pause=0.5, expect='MTI-OK', throw_on_wrong=True)
            self._send(f'MTI+EN', pause=1, expect='MTI-OK', throw_on_wrong=True)
            self.enabled = True
            self.steer((0, 0))

        if type(self.model) is SerialMirror.Optotune:
            self.conn.reset_output_buffer()
            self.conn.reset_input_buffer()
            self._send(f'status', pause=0.5, expect='OK', throw_on_wrong=False)
            self._send(f'acknowledge', pause=0.5, expect='OK', throw_on_wrong=True)
            self.enabled = True
            self.steer((0, 0))

    def disable(self):
        if not self.enabled: return
        self.logger.info('Disabling mirror')
        self.steer((0, 0)) # return to origin
        if type(self.model) is SerialMirror.Mirrorcle:

            # Reset any previous steering commands
            self.conn.reset_output_buffer()
            self.conn.reset_input_buffer()
            for _ in range(3):
                self._send(f'MTI+DI', pause=0, expect='MTI-OK', throw_on_wrong=False)
            self._send(f'MTI+EX', pause=0.5, expect='MTI-Device Exit Command Mode', throw_on_wrong=False)
            self.conn.reset_output_buffer()
            self.conn.reset_input_buffer()
            self.enabled = False
        
        elif type(self.model) is SerialMirror.Optotune:
            self.enabled = False

    def __del__(self):
        self.logger.debug('Mirror is deleted')
        self.disable()