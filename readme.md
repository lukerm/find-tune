# find-tune

## Aim

The objective of this project is to create a program that listens to a continuous stream of sound and identifies when a particular
song - the target track - is playing. This is similar to how home assistants such as Amazon's 'Alexa' function, except they seek out a
different sound (their name). Ultimately, this project will be used to replay the detected positive sound to a speaker, serving as a
doorbell amplifier. 

The track that I'm interested in classifying is committed into this repository in `data/target_tune.wav`. However, there's no reason
you couldn't adapt this to other target tracks that may interest you. To do this, you'd have to train your own model in order to get it
to classify that tune or sound. Going further, this code could be tweaked to create a homemade baby monitor, where the target sound 
would be crying. However, I would recommend _extensive_ testing before deploying on your child!


## How it works

The positive sound, the doorbell, is already saved to `data`, but we also need a collection of other sounds that it might reasonably
come up against. I've used a selection of sounds from the [ontology](https://github.com/audioset/ontology/blob/master/ontology.json) 
created by Google's [AudioSet team](https://ai.google/research/pubs/pub45857) to create our own mini corpus of negative examples for this 
binary classification problem. The sound categories I used are stored in the file `data/non-target_categories.txt`, which is used in
data creation process. 

I use a neural-network based architecture to solve this problem, in particular the "VGGish" network for sound classification (approx. 
600 categories) as a warm start, before fine-tuning to this task. (The unusual name takes inspiration from the network architecture 
designed by the Visual Geometry Group (VGG) of the University of Oxford for their solution to the ImageNet Challenge (computer vision) 
in 2014.) Google have developed the tools for the pre-processing of sound as well as a TensorFlow model, so 
(their project)[https://github.com/tensorflow/models/tree/master/research/audioset] is a dependency of this one. Necessarily, I have
had to derive my own model from theirs which is available to download (see "Installation guide" below). 

There is a lot of cool mathematics connected to sound classification, which you can read more about [here](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/).
I also have to tip my hat in the direction of the [project](https://github.com/devicehive/devicehive-audio-analysis) from DeviceHive
whose code I have adapted for my own production needs (i.e. capturing, processing and classifying sound on the fly). 


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

If you want to train the retrain the model, follow the instructions found in the `train` folder.

If you are going to install this project on a Raspberry Pi, you'll also need to run the instructions in `setup/install_pi3.sh`, which I hope
is a catalogue of all of the extra instructions required (but cannot guarantee it). If you do attempt this, please do get in touch, whether
you succeed or get stuck (e.g. through creating an Issue).

