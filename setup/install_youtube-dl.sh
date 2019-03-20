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

# youtube-dl is a lightweight program that facilitates downloading YouTube audio clips from command line
# Documentation: https://github.com/rg3/youtube-dl/blob/master/README.md#readme
# Note: requires `python` to be available

# Install
sudo apt install curl
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

sudo apt install ffmpeg

# Example: check it works
# This will download a WAV file which you subsequently can play or delete
youtube-dl -x --audio-format "wav" -o 'yt8m_sound_%(id)s.%(ext)s' https://www.youtube.com/watch?v=0fOHh5Q7Q1E
