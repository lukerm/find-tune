#  Copyright (C) 2018 lukerm
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Clone tensorflow's repo (approx 0.5GB)
git clone git@github.com:tensorflow/models.git ~/tf-models
cd ~/tf-models/research/audioset

# Download models / parameters
wget https://storage.googleapis.com/audioset/vggish_model.ckpt # 0.25GB
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz # 70KB

# Perform installation test
python3 vggish_smoke_test.py
