# This bash script walks through the steps required to prepare the audio features (from VGGish model)
# from the sound bites downloaded from YouTube. Please ensure setup/install_youtube-dl.sh has been executed.

# Prepare the list of audio files to download from YouTube
# Output files: data/dl_youtube_links.txt data/link_details.json
python3 prepare_youtube_download.py

# Where possibile, download the audio content only from the links listed in dl_youtube_links.txt
# We ignore errors, as some of these files may have been removed
# Use a random sleep to prevent being blocked
# Note: this can take a long time, depending on how many files you want to download
mkdir ../data/youtube_orig
youtube-dl -x --audio-format "wav" -o '../data/youtube_orig/yt8m_sound_%(id)s.%(ext)s' --ignore-errors --max-filesize 100m --sleep-interval 2 --max-sleep-interval 4 --batch-file ../data/dl_youtube_links.txt #> ../data/dl.log 2>&1

# Report number of downloads
n_dls=$(ll ../data/youtube_orig/*.wav | wc -l)
echo "Number of files downloaded: $n_dls"

# Clip the wav files to correct segment
mkdir ../data/youtube_clip
python3 clip_youtube_files.py

# Clear out the originals, as they are probably quite large in sum
rm ../data/youtube_orig/*.wav

# Use the VGGish model to extract 128-D feature embeddings for each clipped track
# Note: this does not perform the post-processing steps cited here:
# https://github.com/tensorflow/models/blob/master/research/audioset/vggish_postprocess.py
python3 create_embedding_data.py
