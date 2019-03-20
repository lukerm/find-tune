# Copyright (C) 2017 DataArt
# Modifications copyright (C) 2018 lukerm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by lukerm (2018). View the original file at:
#     https://github.com/devicehive/devicehive-audio-analysis/blob/master/audio/device.py

import pyaudio


__all__ = ['AudioDevice']


class AudioDevice(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.sample_rate = int(self.pa.get_default_input_device_info()['defaultSampleRate'])

        self.in_stream = self.pa.open(format=pyaudio.paInt16, channels=1,
                                      rate=self.sample_rate, input=True, frames_per_buffer=self.sample_rate)
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
        # As it's not essential to capture every byte of sound in the buffer,
        # we'll turn off the overflow feature
        return self.in_stream.read(n, exception_on_overflow=False)

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
