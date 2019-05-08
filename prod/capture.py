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
#     https://github.com/devicehive/devicehive-audio-analysis/blob/master/capture.py

import argparse
import threading
import time
import os
import random
import numpy as np

from datetime import date
from scipy.io import wavfile

import logging.config
from log_config import LOGGING

from prod.captor import Captor
from prod.processor import WavProcessor, format_predictions

from definitions import ROOT_DIR, JUKEBOX_DIR, JUKEBOX_CNF


os.makedirs(os.path.join(ROOT_DIR, 'prod', 'logs'), exist_ok=True)
logging.config.dictConfig(LOGGING)
logger = logging.getLogger('audio_analysis.capture')


class Capture(object):
    _ask_data = None
    _captor = None
    _save_path = None
    _processor_sleep_time = 0.01
    _process_buf = None

    def __init__(self, min_time, max_time, path=None, verbose=False):
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError('"{}" doesn\'t exist'.format(path))
            if not os.path.isdir(path):
                raise FileNotFoundError('"{}" isn\'t a directory'.format(path))

        # Ensure weights loaded before anything else happens
        self._processor = WavProcessor()
        logger.info("Network model's weights loaded")
        self._save_path = path
        self._ask_data = threading.Event()
        self._captor = Captor(min_time, max_time, self._ask_data, self._process)
        # Sample rate from the AudioDevice within Captor
        self._sample_rate = self._captor.audio_device.sample_rate
        # Whether to be verbose
        self.verbose = verbose

    def start(self):
        self._captor.start()
        self._process_loop()

    def _process(self, data):
        self._process_buf = np.frombuffer(data, dtype=np.int16)

    def _process_loop(self):
        with self._processor as proc:
            # Confirm that the system is ready with a ding
            os.system('paplay %s' % os.path.join(JUKEBOX_DIR, 'ding.wav'))
            self._ask_data.set()
            while True:
                try:
                    if self._process_buf is None:
                        # Waiting for data to process
                        time.sleep(self._processor_sleep_time)
                        continue

                    self._ask_data.clear()
                    if self._save_path:
                        f_path = os.path.join(
                            self._save_path, 'record_{:.0f}.wav'.format(time.time())
                        )
                        wavfile.write(f_path, self._sample_rate, self._process_buf)
                        logger.info('"{}" saved.'.format(f_path))

                    preds = proc.get_predictions(self._sample_rate, self._process_buf, verbose=self.verbose)
                    logger.info('Target detected: %.0f%% (%d/%d)' % ((100 * preds.mean()), sum(preds), len(preds)))
                    if preds.mean() > 0.25:
                        track_to_play = self.find_playback_track()
                        logger.info('Sounding doorbell: %s' % track_to_play)
                        os.system('paplay %s' % os.path.join(JUKEBOX_DIR, track_to_play))
                    self._process_buf = None
                    self._ask_data.set()

                except Exception:
                    logger.exception('Fatal error in _process_loop')

    def find_playback_track(self):
        # Attempt to access the Easter egg for this day, otherwise choose one of the default tracks at random
        today = date.today().strftime('%m-%d')
        track = JUKEBOX_CNF['easter_eggs'].get(today, random.choice(JUKEBOX_CNF['defaults']))
        return track


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture and process audio')
    parser.add_argument('--min_time', type=float, default=5, metavar='SECONDS',
                        help='Minimum capture time')
    parser.add_argument('--max_time', type=float, default=7, metavar='SECONDS',
                        help='Maximum capture time')
    parser.add_argument('-s', '--save_path', type=str, metavar='PATH',
                        help='Save captured audio samples to provided path',
                        dest='path')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print out extra timing info')


    args = parser.parse_args()
    c = Capture(**vars(args))
    c.start()
