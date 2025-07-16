# Semi-supervised VQA Multi-modal Explanation (SME)
The official implementation of paper: Semi-Supervised VQA Multi-Modal Explanation via Self-Critical Learning

### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- [CLIP](https://github.com/openai/CLIP) (install with `pip install git+https://github.com/openai/CLIP.git`)
- [transformers](https://huggingface.co/docs/transformers/index) (install with `pip install transformers`)
- [accelerate](https://huggingface.co/docs/accelerate/index.html) for distributed training (install with `pip install git+https://github.com/huggingface/accelerate`)


We conduct experiments on 5 different VQA/NLE Datasets: **VQAv2, OKVQA, A-OKVQA, VQA-X and VQA-HAT**. 

### Data Preparation
#### Images Download
<details>
<summary>download</summary>
Please download the COCO images into a folder in your directory named `images` using the following links:
<br>
- [COCO](https://cocodataset.org/#download) `train2014` and `val2014` images<br>
</details>

#### Annotations Download

**VQA-v2:**
<details>
<summary>download</summary>
<br>
- [train-annotation](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip) <br>
- [train-question](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)<br>
- [val-annotation](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)<br>
- [val-question](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)<br>
</details>

**OKVQA:**
<details>
<summary>download</summary>
<br>
- [train](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip) <br>
- [val](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)<br>
</details>

**A-OKVQA:**
<details>
<summary>download</summary>
<br>
- [aokvqa](https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz) <br>
</details>

**VQA-X:**
<details>
<summary>download</summary>
<br>
- [VQA-X](https://drive.google.com/drive/folders/16sJjeEQE2o23G-GGUi870ubXzJjdRDua?usp=sharing)<br>
</details>


**VQA-HAT:**
<details>
<summary>download</summary>
<br>
- [train](http://s3.amazonaws.com/vqa-hat/vqahat_train.zip)<br>
- [val](http://s3.amazonaws.com/vqa-hat/vqahat_val.zip)<br>
</details>


### Running
#### VQA-X
Please run from the command line with: <br>
```bash
accelerate launch vqax_train_with_r.py
```

#### VQA-HAT
Please run from the command line with: <br>
```bash
accelerate launch vqahat_train.py
```
