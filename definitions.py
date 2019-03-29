#  Copyright (C) 2018 lukerm
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Specify the directory to look for playback songs
JUKEBOX_DIR = os.path.join(DATA_DIR, 'jukebox')
# Specify the default song, which can be overridden by an "easter egg",
# indexed by date strings with format MM-DD. A trivial example is shown
# below (for more exciting example, replace it by a different wav file in
# the JUKEBOX_DIR)
JUKEBOX_CNF = {
    'default': 'target_tune.wav',
    'easter_eggs': {
         '12-24': 'jingle_bell_rock.wav',
         '12-25': 'jingle_bell_rock.wav',
    }
}

# This path must match where you cloned tensorflow's models repository to - please adjust as required
VGGISH_DIR = os.path.join(os.path.expanduser('~'), 'tf-models', 'research', 'audioset')
#VGGISH_DIR = os.path.join(ROOT_DIR, '..', 'tf-models', 'research', 'audioset')
