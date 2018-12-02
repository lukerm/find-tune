# This bash script walks through the steps required to prepare the audio features (from VGGish model) 
# from the sound bites downloaded from YouTube. Please ensure install_youtube-dl.sh has been executed.

# Prepare the list of audio files to download from YouTube
# Output files: data/dl_youtube_links.txt data/link_details.json
python3 prepare_youtube_download.py

# Where possibile, download the audio content only from the links listed in dl_youtube_links.txt
# We ignore errors, as some of these files may have been removed
# Use a random sleep to prevent being blocked
mkdir data/youtube_orig
youtube-dl -x --audio-format "wav" -o 'data/youtube_orig/yt8m_sound_%(id)s.%(ext)s' --ignore-errors --max-filesize 100m --sleep-interval 2 --max-sleep-interval 4 --batch-file data/dl_youtube_links.txt #> data/dl.log 2>&1

