## Training pipeline

Before training a model, we'll need to prepare the dataset, including downloading sound clips from Google's AudioSet, a corpus of labelled YouTube videos. 
To achieve this, you'll need to install software that could retrieve partial YouTube clips - follow the steps in `../setup/install_youtube-dl.sh`. You can 
read more details about this dataset [here](https://research.google.com/audioset/).

After that:

* `create_youtube_dataset.sh`, which runs the following: 
	* `prepare_youtube_download.py`
	* `youtube-dl` (to download YouTube files)
	* `clip_youtube_files.py`
	* `create_embedding_data.py`
* `models_with_embedding.py` 
* `fine_tune.py`

The first step will download the non-target clips from YouTube and converts them to embedding data for training down-stream models. It takes approximately 
1 hour to run (download-speed dependent, ~4MB/s for me). If you don't want to wait, you can download the YouTube clips and embedding data from AWS: 
`s3://lukerm-ds-open/find-tune/data/youtube_clip/` and `s3://lukerm-ds-open/find-tune/data` respectively.

TODO: note that a linear classifier is nearly on par with dense neural network, but is slightly weaker on the cross-validation test: see `_output.txt`. 
      Plus, we can easily attach that network head to the original VGGish architecture. 
