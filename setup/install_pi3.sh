# Logging steps I found when installing on Raspberry Pi 3 (Raspian Stretch)

# Set up microphone
# https://iotbytes.wordpress.com/connect-configure-and-test-usb-microphone-and-speaker-with-raspberry-pi/
# Please check configuration before copying blindly
arecord -l

sudo apt-get install libportaudio2 portaudio19-dev
sudo apt-get install libatlas-base-dev

sudo apt-get install cmake
sudo apt-get install libedit-dev

# Install llvmlite from source (it's a bit painful, but steps mapped out below)
# Using install guide: https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html
cd
wget releases.llvm.org/7.0.0/llvm-7.0.0.src.tar.xz
tar xf llvm-7.0.0.src.tar.xz
rm llvm-7.0.0.src.tar.xz
git clone git@github.com:numba/llvmlite
cd llvm-7.0.0.src
patch -p1 < ~/llvmlite/conda-recipes/D47188-svml-VF.patch
patch -p1 < ~/llvmlite/conda-recipes/partial-testing.patch
patch -p1 < ~/llvmlite/conda-recipes/twine_cfg_undefined_behavior.patch
patch -p1 < ~/llvmlite/conda-recipes/llvmdev/0001-RuntimeDyld-Fix-a-bug-in-RuntimeDyld-loadObjectImpl-.patch 
patch -p1 < ~/llvmlite/conda-recipes/llvmdev/0001-RuntimeDyld-Add-test-case-that-was-accidentally-left.patch
cd
cd llvm-7.0.0.src
export PREFIX=/usr/bin CPU_COUNT=2
chmod +x ~/llvmlite/conda-recipes/llvmdev/build.sh
sudo mkdir /usr/bin/include
sudo ~/llvmlite/conda-recipes/llvmdev/build.sh # This can take up to 2hrs
cd ~/llvmlite
export LLVM_CONFIG=$(which llvm-config)
sudo python3 setup.py build
python3 runtests.py
pip3 install --upgrade .

