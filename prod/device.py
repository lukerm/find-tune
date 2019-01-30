## Edited from: https://github.com/devicehive/devicehive-audio-analysis/blob/master/audio/device.py

import pyaudio


__all__ = ['AudioDevice']


class AudioDevice(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.sample_rate = int(self.pa.get_default_input_device_info()['defaultSampleRate'])

        self.in_stream = self.pa.open(format=pyaudio.paInt16, channels=1,
                                      rate=self.sample_rate, input=True)
        self.in_stream.start_stream()
        self.out_stream = self.pa.open(format=pyaudio.paInt16, channels=1,
                                       rate=self.sample_rate, output=True)
        self.out_stream.start_stream()

    def close(self):
        self.in_stream.close()
        self.out_stream.close()
        self.pa.terminate()

    def write(self, b):
        return self.out_stream.write(b)

    def read(self, n):
        return self.in_stream.read(n)

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
