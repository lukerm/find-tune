# Install 
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
# Example: check it works
youtube-dl -x --audio-format "wav" -o 'yt8m_sound_%(id)s.%(ext)s' https://www.youtube.com/watch?v=0fOHh5Q7Q1E






