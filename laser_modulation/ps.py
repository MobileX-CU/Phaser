import serial
import time
import logging

# https://www.rohde-schwarz.com/webhelp/HMPSeries_HTML_UserManual_en/HMPSeries_HTML_UserManual_en.htm
class HMP4040(object):

    def __init__(self, port=None):
        self.logger = logging.getLogger()
        self.conn = serial.Serial(port, 115200, timeout=1) if port else None
        self.reset()

    def _send(self, cmd, endl='\n', delay=0.1):
        cmd = (cmd + endl).encode('utf-8')
        self.logger.debug(f'> {cmd}')
        if self.conn: self.conn.write(cmd)
        time.sleep(delay)

    def reset(self):
        self.disable_out()
        self.logger.info('Resetting PS')
        for ch in range(4):
            self._send(f'INST OUT{ch+1}')
            self._send(f'OUTP:SEL 0')
            self._send(f'OUTP OFF')
            self._send(f'ARB:STOP 1')
            self._send(f'ARB:CLEAR 1')

    def enable_ch(self, ch):
        self._send(f'INST OUT{ch}')
        self._send(f'OUTP:SEL 1')
        self._send(f'OUTP ON')

    def disable_ch(self, ch):
        self._send(f'INST OUT{ch}')
        self._send(f'OUTP:SEL 0')
        self._send(f'OUTP OFF')

    def enable_out(self):
        self.logger.info('Enabling PS output')
        self._send('OUTP:GEN 1')

    def disable_out(self):
        self.logger.info('Disabling PS output')
        self._send('OUTP:GEN 0')

    def ch_set_dc(self, ch, I=0, V=0):
        self.logger.info(f'Setting ch={ch} for DC output -> {I}A@{V}V')
        self._send(f'INST OUT{ch}')
        self._send(f'VOLT {V}')
        self._send(f'CURR {I}')
        self._send(f'OUTP:SEL 1')
        self._send(f'OUTP ON')
    
    def ch_unset(self, ch):
        self.logger.info(f'Unsetting ch={ch}')
        self._send(f'INST OUT{ch}')
        self._send(f'OUTP:SEL 0')
        self._send(f'OUTP OFF')
        self._send(f'ARB:STOP 1')
        self._send(f'ARB:CLEAR 1')

    # Caveats:
    # 1. Minimum delay time is 0.01s ==> 50hz, 25hz
    # 2. Only allows 2 digits to be sent after the decimal point
    # 3. There is a 10ms delay between sequences
    def ch_set_ac_square(self, ch, f, VHigh, IHigh, VLow, ILow):
        self.logger.info(f'Setting ch={ch} for AC square wave output')
        t0 = round(max(1/(2*f), 0.01), 2)
        t1 = round(max(1/(2*f), 0.01), 2)
        self.logger.info(f'{f}Hz -> {IHigh}A@{VHigh}V for {t0}s, {ILow}A@{VLow}V for {t1}s')
        seq = f'{VHigh},{IHigh},{t0:.2f},{VLow},{ILow},{t1:.2f}'
        seq = ((',' + seq) * 63)[1:]
        self._ch_set_ac(ch, seq)

    def _ch_set_ac(self, ch, seq):
        self._send(f'INST OUT{ch}')
        self._send(f'ARB:DATA {seq}')
        self._send(f'ARB:TRAN {ch}')
        self._send(f'ARB:STAR {ch}')
        self._send(f'OUTP ON')