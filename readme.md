# find-tune

## Aim

The objective of this project is to create a program that listens to a continuous stream of sound and identifies when a particular
song - the target track - is playing. This is similar to how home assistants such as Amazon's 'Alexa' function, except they seek out a
different sound (their name). Ultimately, this project will be used to replay the detected positive sound to a speaker, serving as a
doorbell amplifier. 

The track that I'm interested in classifying is committed into this repository in `data/target_tune.wav`. However, there's no reason
you couldn't adapt this to other target tracks that may interest you. To do this, you'd have to train your own model in order to get it
to classify that tune or sound. Going further, this code could be tweaked to create a homemade baby monitor, where the target sound 
would be crying (_extensive_ testing would be required before deploying on a child!).


## How it works

The positive sound, the doorbell, is already saved to `data`, but we also need a collection of other sounds that it might reasonably
come up against. I've used a selection of sounds from the [ontology](https://github.com/audioset/ontology/blob/master/ontology.json) 
created by Google's [AudioSet team](https://ai.google/research/pubs/pub45857) to create our own mini corpus of negative examples for this 
binary classification problem. The sound categories I used are stored in the file `data/non-target_categories.txt`, which is used in
the data creation process. 

I use a neural-network based architecture to solve this problem, in particular the "VGGish" network for sound classification (approx. 
600 categories) as a warm start, before fine-tuning to this task. (The unusual name takes inspiration from the network architecture 
designed by the Visual Geometry Group (VGG) of the University of Oxford for their solution to the ImageNet Challenge (computer vision) 
in 2014.) Google have developed the tools for the pre-processing of sound as well as a TensorFlow model of VGGish, so 
[their project](https://github.com/tensorflow/models/tree/master/research/audioset) is a dependency of this one. Necessarily, I have
had to derive my own model from theirs which is available to download (see "Installation guide" below). 

There is a lot of cool mathematics connected to sound classification, which you can read more about [here](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/).
I also have to tip my hat in the direction of the [project](https://github.com/devicehive/devicehive-audio-analysis) from DeviceHive
whose code I have adapted for my own production needs (i.e. capturing, processing and classifying sound on the fly). 


## Installation guide

To run this project, you might want to a) train and fine-tune the model yourself or b) install onto a Raspberry Pi (or both!).
In the first case, you will not be able to fine-tune the network on a Raspberry Pi as the memory requirements are higher than
the device allows (as of third generation hardware), but an ordinary laptop or PC should suffice. In the latter case, you do 
not have to perform the training yourself, as the fine-tuned model can be downloaded from the AWS key: `s3://lukerm-ds-open/find-tune/data/`.

In both cases, please follow these instructions to get set up, bearing in mind that some of the python requirements need special attention
for the Raspberry Pi:

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


## Production

Once you have completed the installation instructions, either for your laptop or Pi, your device should be production-ready.
Please ensure that your (USB) microphone is turned up. As a first step, you can run `prod/test_load_model.py` to check that the 
TensorFlow model loads correctly through the Keras interface. 

On a Raspberry Pi, this will fail if you do not have enough swap memory allocated (1GB should be sufficient when
[resizing the swap](https://www.bitpi.co/2015/02/11/how-to-change-raspberry-pis-swapfile-size-on-rasbian/)).
This is because loading the model's weights is a very memory-intensive process requiring more than the 1GB of RAM available
on the third-generation device. Even then, it will take several minutes to load the model, and will appear frozen during
that time, but patience will prevail! Once it has everything loaded you will get a message and the program will exit.

After that, I recommend moving onto the `prod/capture.py` file which loads the model before capturing, processing and
classifying the sound coming through the microphone. It will do that in chunks of about five seconds (configurable), printing its
predictions to the terminal. In addition, if it detects the target track with sufficient confidence, then it will play it back
through the system's main audio output. (This is actually configurable as you can play a different track if you want, or even different songs
on different days - see `definitions.py` for more details.) PulseAudio's `paplay` is required for this, so please make sure to follow the
instructions in `setup/install_pi3.sh` (it may already be available on larger Linux distros, such as Ubuntu).

The default audio sink on a Raspberry Pi will not be audible, so you'll have to re-route via Bluetooth. Find the MAC address of the intended
playback device and place it in `prod/check_connected.sh`, replacing the *'s. Then run that script through to get it to connect (ensure
the device is on and that it is not paired to any other device). You may have to run it twice. I had a little difficulty getting my bluetooth
device to switch the high-fidelity "A2DP" mode, and I found that disabling, then re-enabling, the bluetooth service got around this issue,
which is what `check_connected.sh` attempts to do. If that script doesn't work for you, try first establishing the connection manually
using [`bluetoothctl`](https://docs.ubuntu.com/core/en/stacks/bluetooth/bluez/docs/reference/pairing/outbound.html).

You can make this fully automated from boot by copying the lines of `prod/crontab` into your Pi's main `crontab` file.
