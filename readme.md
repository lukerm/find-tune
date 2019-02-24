reference:
  - ontology.json downloaded from https://github.com/audioset/ontology/raw/master/ontology.json
  - https://github.com/tensorflow/models/blob/master/research/audioset/vggish_params.py
  - https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/


## Installation guide

To run this project, you might want to a) train and fine-tune the model yourself or b) install onto a Raspberry Pi (or both!).
In the first case, you will not be able to fine-tune the network on a Raspberry Pi as the memory requirements are higher than
the device allows (as of third generation hardware), but an ordinary laptop or PC should suffice. In the latter case, you do 
not have to perform the training yourself, as the fine-tuned model can be downloaded from the AWS key: `s3://lukerm-ds-open/find-tune/data/`.

In both cases, please follow these instructions to get set up (bearing in mind that some of the python requirements need special attention
for the Raspberry Pi):

* `git clone git@github.com:lukerm/find-tune ~/find-tune/` 
* `cd ~/find-tune/`
* `sudo apt install libportaudio2 portaudio19-dev`
* `pip install --user -r setup/requirements.txt`^
* `setup/install_tf_models.sh` 
* `export PYTHONPATH=$PYTHONPATH:~/find-tune/:~/tf-models/research/audioset/` (also set in `~/.profile`)

^`tensorflow` and `resampy` require special instructions to install on Raspbian - please see the comments _before_ running.

If you are going to install this project on a Raspberry Pi, you'll also need to run the instructions in `setup/install_pi3.sh`, which I hope
is a catalogue of all of the extra instructions required (but cannot guarantee it). 

If you want to train the retrain the model, follow the instructions found in the `train` folder.
