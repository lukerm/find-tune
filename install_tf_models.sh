# Requirements
sudo python3 -m pip install numpy scipy
sudo python3 -m pip install resampy tensorflow six
sudo python3 -m pip install soundfile

# Clone tensorflow's repo (approx 0.5GB)
git clone git@github.com:tensorflow/models.git ~/tf-models
cd ~/tf-models/research/audioset

# Download models / parameters
wget https://storage.googleapis.com/audioset/vggish_model.ckpt # 0.25GB
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz # 70KB

# Perform installation test
python3 vggish_smoke_test.py
