#!/usr/bin/python
import visa

#  to make the file executable:
#  chmod +x script[.py]

# Must pip import pyvisa-py (dev version) and not py-visa (stable version)
# http://pyvisa.readthedocs.io/en/stable/names.html


class Neaspec:
    def __init__(self, ip_address='168.192.2.1', socket='4044'):
    # def __init__(self, ip_address):
        try:
            self.rm = visa.ResourceManager("@py")
            self.instrument_name = "neaspec"
            self.y_counter = 0
            self.connected = False
            ip = "TCPIP0::" + ip_address + "::" + socket + "::SOCKET"
            # ip = "TCPIP0::" + ip_address
            self.my_instrument = self.rm.open_resource(ip, read_termination='\r')
            instrument_id = self.my_instrument.query("*IDN?")
            print "Connected to " + instrument_id + " over TCP/IP"
            if not instrument_id == -1:
                self.connected = True
            reply = self.my_instrument.query("*RST")
            if reply == "\nok":
                print "Neaspec initialized and reset successfully"
        except Exception, e:
            print "Error - unable to connect to Neaspec: {}".format(str(e))


    def LoadPresets(self, params=[]):
            try:
                pulseWidth = params["PWnsec"]  # [nsec]
                frequency = params["freqKHz"]  # [kHz]

                period = 1 / frequency / 1000

                self.my_instrument.write(":pulse1:width {}\r".format(pulseWidth * 1e-9))
                self.my_instrument.write(":pulse2:delay {}\r".format(0.))
                self.my_instrument.write(":pulse1:state {}\r".format(params["CH1"]))

                self.my_instrument.write(":pulse2:width {}\r".format(pulseWidth * 1e-9))
                self.my_instrument.write(":pulse2:delay {}\r".format(period / 2))
                self.my_instrument.write(":pulse2:state {}\r".format(params["CH2"]))

                self.my_instrument.write(":spulse:period {}\r".format(period))
                self.my_instrument.write(":spulse:trig:edge rising\r")
                self.my_instrument.write(":spulse:trig:lev 2.0 V\r")
                self.my_instrument.write(":spulse:trig:mod trig\r")
                self.my_instrument.write(":spulse:mode normal\r")
                self.my_instrument.write(":spulse:state ON\r")

                self.my_instrument.write("*TRG\r")

            except Exception, e:
                print "Error - unable load Neaspec preset <{}>: {}".format(preset, str(e))


    def moveRight(self):
        try:
            # move y- string
            self.my_instrument.write('''}doeyE@@jN3's|
                                    "]o:RO4R::Call
                                    :    @nme:motor_move:    @oidi    :    @tidi:    @arg[[[I"SY:EF[f0.0001:    @blkF''')
            # stop string
            self.my_instrument.write('''}doeyE~@@jN5`'8s
                                        #S wFo:RO4R::Call
                                        :    @nme:motor_move:    @oidi    :    @tidi:    @arg[F:    @blkF''')
            self.y_counter += 1

        except Exception, e:
            print "Error - unable move right preset <{}>: {}".format(preset, str(e))


    def newLine(self):
        try:
            # return to the left corner
            while self.y_counter > 0:
                # self.my_instrument.write("")    # move x+ package
                self.my_instrument.write("")    # move y+ package
                self.my_instrument.write("")    # stop string
                self.y_counter -= 1
            # move down and start a new line
            self.my_instrument.write("")  # move x+ package
            self.my_instrument.write("")  # stop string
            print "Strarted a new line"

        except Exception, e:
            print "Error - unable move start a new line preset <{}>: {}".format(preset, str(e))


if __name__ == "__main__":
    neaspec = Neaspec()
    neaspec.moveRight()
    # neaspec.newLine(count_to_left)

