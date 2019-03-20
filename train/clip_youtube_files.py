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

## Imports ##

import os
import json


## Main ##

with open('../data/link_details.json', 'r') as j:
    link_details = json.load(j)

n_success, n_failure = 0, 0
for cat, urls in link_details.items():
    for url, yt_id, start, end in urls:

        # Required for the -to arg in ffmpeg
        duration = int(end) - int(start)
        assert duration > 0 # Usually 10s

        orig_fname = os.path.join('..', 'data', 'youtube_orig', 'yt8m_sound_%s.wav' % yt_id)
        clip_fname = os.path.join('..', 'data', 'youtube_clip', 'yt8m_sound_%s.wav' % yt_id)
        status = os.system('ffmpeg -loglevel quiet -ss %d -i %s -to %d -c copy %s' % (int(start), orig_fname, duration, clip_fname))

        if status != 0:
            n_failure += 1
            print("FAILURE: %s (doesn't exist)" % yt_id)
        else:
            n_success += 1
            print("SUCCESS: %s" % yt_id)

print('Number of files clipped: %d' % n_success)
print('Number of failed files:  %d' % n_failure)
