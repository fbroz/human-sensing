#!/usr/bin/env python
import collections
import yarp



class MaryTTS:
    #Class for controlling iCub simulator via its RPC port.

    def __init__(self):
        self._port = yarp.Port()
        self._port_name = "/MaryTTS-" + str(id(self)) + "/speechin"
        self._port.open(self._port_name)
        yarp.Network.connect(self._port_name,"/MSpeak/text:i")


    def _execute(self, cmd):
        self._port.write(cmd)

    def speak(self, txt):
        cmd = yarp.Bottle()
        cmd.addString(txt)
        self._execute(cmd)
        
    def __del__(self):
        try:
            self._port.close()
            del self._port
        except AttributeError:
            pass



def main():
    yarp.Network.init() # Initialise YARP
    s = MaryTTS()
    print "sending speech"
    s.speak("Hello, my name is Nikita. I was built in Genoa in Italy, but now I live in Edinburgh in Scotland. I am part of the Heriot Watt robotics laboratory. In this laboratory I work together with many researchers. We are doing fun experiments and games and I learn a lot from them. One of my favorite games is the to learn the shape and names of new objects and toys. Usually to play this game one of the humans working with me shows me an object and tells me its name. I then repeat the name and hopefully the next time I am shown many different objects, I can associate the right object with the right name. Most of the time in the lab I am watching the others working on new programs and applications for me. I am always very excited when I am given a new ability. Yesterday for example I learned how to move my lips correctly when I speak. This is not easy for me and I am the only one in my family who can actually do this. All the other iCub robots don't have a jaw they can move. This makes me very special. I hope soon I will be able to have long conversations with people. Humans have so many interesting things to say and I cannot wait to learn more from them.")
    print "speech sent"
    
    
if __name__ == "__main__":
    main()
