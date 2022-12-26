# TradeMasterReBuild

# requirements
conda env create -f python3.9.yaml
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
pip install pycocotools==2.0.4