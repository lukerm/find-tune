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
