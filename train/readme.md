## Training pipeline

Before training a model, we'll need to prepare the dataset, including downloading sound clips from Google's AudioSet, a corpus of labelled YouTube videos. 
To achieve this, you'll need to install software that can retrieve partial YouTube clips - follow the steps in `../setup/install_youtube-dl.sh` to do so. 
You can read more details about the dataset [here](https://ai.google/research/pubs/pub45857).

After that, run:

1. `create_youtube_dataset.sh`, which runs the following automatically: 
	* `prepare_youtube_download.py`
	* `youtube-dl` (to download YouTube files)
	* `clip_youtube_files.py`
	* `create_embedding_data.py`
2. `models_with_embedding.py` 
3. `fine_tune.py`

The first step will download the non-target clips from YouTube and convert them to embedding data for training down-stream models. It takes approximately 
one hour to run (download-speed dependent, ~4MB/s for me). If you don't want to wait, you can download the YouTube clips and embedding data from AWS: 
`s3://lukerm-ds-open/find-tune/data/youtube_clip/` and `s3://lukerm-ds-open/find-tune/data/embedding_data.npz` respectively. When you have these datasets, 
you can get onto the cool Data Science stuff.

The second step performs some exploratory modelling of the binary task, using the 128-D embedding features output from the penultimate layer of VGGish
and a variety of model types. The output of this script is stored in `models_with_embedding_output.txt`. I found that both a linear classifier and 
shallow, dense neural network performed well, although the network was marginally better on the more rigorous cross-validation test. As the network 
is easier to attach to the original architecture, we'll proceed with that. 

(Note: to give a flavour of how the neural network head makes its predictions, I have provided a visualization of target-track data points before and
after running them through the classification task: see more details in the notebook `visualize_embedding_data.ipynb`.)

The third step is used to attach the trained model head to the VGGish architecture and fine tune the weights to produce the final model. The output of running
this script can be found in `fine_tune_output.txt`. The model is stored in the AWS key `s3://lukerm-ds-open/find-tune/data/my_vggish_network.[h5|json]`.
