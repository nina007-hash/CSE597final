
## 1. Environments
`conda create -n train_cbart python==3.7`
`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
`pip install transformers==3.0.2`
`pip install pympler==0.8`

If have ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with LibreSSL 2.8.3. See: https://github.com/urllib3/urllib3/issues/2168
`pip install urllib3==1.26.6`


## 2. Prepare Data

`cd data && mkdir Flickr30k`
download Flickr30k dataset(only captions needed) from https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
here is [[data/Flickr30k/results.csv]]

split it to train.txt and dev.txt:
`python process2txt.py`

Create synthetic data:

`cd utils`
`python convert_txt_to_pt.py --dataset Flickr30k`
`python create_synthetic_data.py --max_insert_label 1 --insert_mode 0 --dataset Flickr30k`

## 3. Train

Dowload pretrianed model(CBART-large for One-Billion-Word) from https://drive.google.com/file/d/13NOAsdSnO-eLIDxdo0M-_sX2KxyrYndX/view?usp=sharing
put it to `checkpoints-pretrained`

`cd models`
`python bart.py --batch_size 25 --gpu 0 --dataset Flickr30k --bart large --epochs 2`


