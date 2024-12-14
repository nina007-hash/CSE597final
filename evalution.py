
import os
from glob import glob

import json
from pycocoevalcap.cider.cider import Cider


SCORER = Cider()


# domain is ["in", "out", "near"]
def compute_cider_nocaps(gt_root, predict_jsonf, domain="in"):
    # 1. load predict caption
    res = {}
    with open(predict_jsonf, "r") as f:
        data = json.load(f)

        for k, v in data.items():
            res[k] = [v]
        
        del data

    # 2. get gts
    gts = {}
    for k, v in res.items():
        gt_caption_file = os.path.join(gt_root, f"{domain}-domain", k + ".txt")
        assert os.path.exists(gt_caption_file)
        with open(gt_caption_file, "r", encoding="utf=8") as f:
            lines = f.readlines()
            captions = []
            for line in lines:
                _caption = line.strip()
                if _caption:
                    captions.append(_caption)
            gts[k] = captions

    # 3. compute CIDEr score
    score, scores = SCORER.compute_score(gts, res)
    return score * 100, gts, res



def compute_cider_coco(gt_root, predict_jsonf):
    # 1. load predict caption
    res = {}
    with open(predict_jsonf, "r") as f:
        data = json.load(f)

        for k, v in data.items():
            res[k] = [v]
        
        del data

    # 2. get gts
    gts = {}
    for k, v in res.items():
        gt_caption_file = os.path.join(gt_root, k + ".txt")
        assert os.path.exists(gt_caption_file)
        with open(gt_caption_file, "r", encoding="utf=8") as f:
            lines = f.readlines()
            captions = []
            for line in lines:
                _caption = line.strip()
                if _caption:
                    captions.append(_caption)
            gts[k] = captions

    # 3. compute CIDEr score
    score, scores = SCORER.compute_score(gts, res)
    return score * 100



if __name__ == "__main__":


    # 0. CBART_Flikr30k nocaps
    CBART_Flickr_CIDEr_scores = []
    overall_gts = {}
    overall_res = {}
    for domain in ["in", "near", "out"]:
        score, gts, res = compute_cider_nocaps(
            gt_root="DownloadDatasets/nocaps/val_captions", 
            predict_jsonf=f"DownloadDatasets/nocaps/outputs/CBART_Flikr30k/{domain}-domain/MeaCap_{domain}-domain_memory_flickr30k_lmTrainingCorpus_CBART_Flikr30k_0.1_0.8_0.2_k200.json", 
            domain=domain
        )
        CBART_Flickr_CIDEr_scores.append(score)
        overall_gts.update(gts)
        overall_res.update(res)
    
    score, scores = SCORER.compute_score(overall_gts, overall_res)
    CBART_Flickr_CIDEr_scores.append(score * 100)


    # 0.1. CBART_Flikr30k COCO2014
    CBART_Flickr30k_COCO_score = compute_cider_coco(
        gt_root="DownloadDatasets/COCO2014/Karpathy-splitted-captions",
        predict_jsonf="DownloadDatasets/COCO2014/outputs/CBART_Flikr30k/MeaCap_Karpathy-splitted-test_memory_flickr30k_lmTrainingCorpus_CBART_Flikr30k_0.1_0.8_0.2_k200.json"
    )


    # 1. CBART_cc3m nocaps
    CBART_cc3m_CIDEr_scores = []
    overall_gts = {}
    overall_res = {}
    for domain in ["in", "near", "out"]:
        score, gts, res = compute_cider_nocaps(
            gt_root="DownloadDatasets/nocaps/val_captions", 
            predict_jsonf=f"DownloadDatasets/nocaps/outputs/CBART_cc3m/{domain}-domain/MeaCap_{domain}-domain_memory_cc3m_lmTrainingCorpus_CBART_cc3m_0.1_0.8_0.2_k200.json", 
            domain=domain
        )
        CBART_cc3m_CIDEr_scores.append(score)
        overall_gts.update(gts)
        overall_res.update(res)
    
    score, scores = SCORER.compute_score(overall_gts, overall_res)
    CBART_cc3m_CIDEr_scores.append(score * 100)

    # 2. CBART_COCO nocaps
    CBART_COCO_CIDEr_scores = []
    overall_gts = {}
    overall_res = {}
    for domain in ["in", "near", "out"]:
        score, gts, res = compute_cider_nocaps(
            gt_root="DownloadDatasets/nocaps/val_captions", 
            predict_jsonf=f"DownloadDatasets/nocaps/outputs/CBART_COCO/{domain}-domain/MeaCap_{domain}-domain_memory_coco_lmTrainingCorpus_CBART_COCO_0.1_0.8_0.2_k200.json", 
            domain=domain
        )
        CBART_COCO_CIDEr_scores.append(score)
        overall_gts.update(gts)
        overall_res.update(res)
    
    score, scores = SCORER.compute_score(overall_gts, overall_res)
    CBART_COCO_CIDEr_scores.append(score * 100)

    # print NoCaps val (CIDEr)
    print("                 NoCaps val (CIDEr):")
    print("              In  / Near / Out  / Overall")
    print("CBART_cc3m:  {:.1f} / {:.1f} / {:.1f} / {:.1f}".format(*CBART_cc3m_CIDEr_scores))
    print("CBART_COCO:  {:.1f} / {:.1f} / {:.1f} / {:.1f}".format(*CBART_COCO_CIDEr_scores))
    print("CBART_F30k:  {:.1f} / {:.1f} / {:.1f} / {:.1f}".format(*CBART_Flickr_CIDEr_scores))


    # 3. CBART_cc3m COCO2014
    CBART_cc3m_COCO_score = compute_cider_coco(
        gt_root="DownloadDatasets/COCO2014/Karpathy-splitted-captions",
        predict_jsonf="DownloadDatasets/COCO2014/outputs/CBART_cc3m/MeaCap_Karpathy-splitted-test_memory_cc3m_lmTrainingCorpus_CBART_cc3m_0.1_0.8_0.2_k200.json"
    )

    # 4. CBART_COCO COCO2014
    CBART_COCO_COCO_score = compute_cider_coco(
        gt_root="DownloadDatasets/COCO2014/Karpathy-splitted-captions",
        predict_jsonf="DownloadDatasets/COCO2014/outputs/CBART_COCO/MeaCap_Karpathy-splitted-test_memory_coco_lmTrainingCorpus_CBART_COCO_0.1_0.8_0.2_k200.json"
    )

    print("\n===================\n")
    print("CBART_cc3m (COCO2014 test) :  {:.1f}".format(CBART_cc3m_COCO_score))
    print("CBART_COCO (COCO2014 test) :  {:.1f}".format(CBART_COCO_COCO_score))
    print("CBART_F30k (COCO2014 test) :  {:.1f}".format(CBART_Flickr30k_COCO_score))


"""Results:
                 NoCaps val (CIDEr):
              In  / Near / Out  / Overall
CBART_cc3m:  23.4 / 24.3 / 31.2 / 26.4
CBART_COCO:  40.6 / 41.8 / 37.8 / 40.8
CBART_Flickr30k:  45.4 / 31.8 / 18.0 / 29.0
CBART_cc3m (MSCOCO CIDEr) :  25.3
CBART_COCO (MSCOCO CIDEr) :  54.3
"""
