
# Evalution for MeaCap: Memory-Augmented Zero-shot Image Captioning

## 1. Create conda environment
conda create -n meacap_test python=3.10
conda activate meacap_test

## 2. Install requirements 
install pytorch first: 
`pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`
[notice] Eidt requirements.txt line4: # torch~=2.0.0+cu118

install others:
`pip install -r requirements.txt`
`pip install -U sentence-transformers==2.6.1`

## 3. download datasets and karpathy-split for coco
(1) COCO2014 datasets
download COCO2014 from: https://cocodataset.org/#download

download karpathy-split from: http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip it to ./DownloadDatasets/COCO2014/Karpathy-split

cd DownloadDatasets/COCO2014
run `python process.py`  (will get 5000 `test` images by karpathy-split)

(2) nocaps datasets
download `nocaps_val_4500_captions.json` from https://nocaps.org/download  (direct link: https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json)

cd DownloadDatasets/nocaps
run `python download.py`


## 4.Get Predict Results:
Tips: prompts are needless.

### 4.1 test on nocaps datasets

#### 4.1.1 CBART_COCO (in/out/near domain)
python inference.py  --memory_id coco --lm_model_path ./checkpoints/CBART_COCO --img_path ./DownloadDatasets/nocaps/val/in-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_COCO/in-domain
python inference.py  --memory_id coco --lm_model_path ./checkpoints/CBART_COCO --img_path ./DownloadDatasets/nocaps/val/out-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_COCO/out-domain
python inference.py  --memory_id coco --lm_model_path ./checkpoints/CBART_COCO --img_path ./DownloadDatasets/nocaps/val/near-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_COCO/near-domain

#### 4.1.2 CBART_cc3m (in/out/near domain)
python inference.py --memory_id cc3m --memory_caption_path data/memory/cc3m/memory_captions.json --lm_model_path ./checkpoints/CBART_cc3m --img_path ./DownloadDatasets/nocaps/val/in-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_cc3m/in-domain
python inference.py --memory_id cc3m --memory_caption_path data/memory/cc3m/memory_captions.json --lm_model_path ./checkpoints/CBART_cc3m --img_path ./DownloadDatasets/nocaps/val/out-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_cc3m/out-domain
python inference.py --memory_id cc3m --memory_caption_path data/memory/cc3m/memory_captions.json --lm_model_path ./checkpoints/CBART_cc3m --img_path ./DownloadDatasets/nocaps/val/near-domain --output_path ./DownloadDatasets/nocaps/outputs/CBART_cc3m/near-domain


### 4.2 test on COCO2014-test(karpathy-split) datasets

#### 4.2.1 CBART_COCO
python inference.py --memory_id coco --lm_model_path ./checkpoints/CBART_COCO --img_path ./DownloadDatasets/COCO2014/Karpathy-splitted-test --output_path ./DownloadDatasets/COCO2014/outputs/CBART_COCO

#### 4.2.2 CBART_cc3m
python inference.py --memory_id cc3m --memory_caption_path data/memory/cc3m/memory_captions.json --lm_model_path ./checkpoints/CBART_cc3m --img_path ./DownloadDatasets/COCO2014/Karpathy-splitted-test --output_path ./DownloadDatasets/COCO2014/outputs/CBART_cc3m


## 5. Evalution
`pip install pycocoevalcap`
`python evalution.py`

