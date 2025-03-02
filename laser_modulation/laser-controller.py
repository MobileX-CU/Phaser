import time
import logging
import sys
import ps
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

# Setup output file of command streams to compare against XYZT stream
t = time.strftime('%Y%m%d_%H%M')
fname = f'command-stream/{t}.csv'
with open(fname, 'a') as file:

    ch = 1 #channel of laser on power supply
    ledch = 2 #channel of LEDs
    V = 3 # laser voltage (cap at 3 V for safety, and modulate the current)
    Ih = 1.8 # modulation high current [1.8A]
    Il = 1.1 # modulation low current (for going forward with DC output) [1.1A]
    f = None # DC, otherwise modulated at f [Hz]
    exit = False

    # Initialize PS
    hmp = ps.HMP4040('/dev/cu.usbmodemVCP1083731')
    hmp.ch_unset(ch) # reset laser from any previous settings
    hmp.enable_ch(ledch) # always enable LEDs
    hmp.enable_out() # enable all power supply outputs
    while not exit:
        print(f)
        if f is None:
            hmp.ch_set_dc(ch=ch, I=Ih, V=V)
        else:
            hmp.ch_set_ac_square(ch=ch, f=f, VHigh=V, IHigh=Ih, VLow=V, ILow=Il)
        file.write(f'{time.time()},{f}\n')
        file.flush()
        hmp.enable_ch(ch)
        try:
            while not exit: time.sleep(1)
        except KeyboardInterrupt:
            hmp.disable_ch(ch) # disable laser only
            hmp.ch_unset(ch)
            cmd = input('Enter command (l|r|f|any)')
            if cmd == 'l':
                f = 12.5
            elif cmd == 'r':
                f = 25
            elif cmd == 'f':
                f = None
            else:
                exit = True

    # Cleanup
    hmp.ch_unset(ch)
    hmp.disable_out()
