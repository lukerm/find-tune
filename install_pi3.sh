# Logging issues I found during set up on Raspberry Pi 3 (Raspian Stretch)

## h5py
sudo apt-get update
sudo apt-get install -y libhdf5-dev
sudo apt-get update
sudo apt-get install -y libhdf5-serial-dev
pip3 install h5py
python3 -c "import h5py; print('success')"
