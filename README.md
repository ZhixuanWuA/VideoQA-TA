# MViQA-DOTS
Multi-modal video question-answering
<p align="left">
    <img src="https://github.com/ZhixuanWuA/MViQA-DOTS/blob/main/ICONS.png" width="150" style="margin-bottom: 0.2;"/>
<p>


## 🛠️ Requirements and Installation
Basic Dependencies:
* Python >= 3.8
* Pytorch >= 1.10.0
* CUDA Version >= 11.8
* transformers >= 4.28.0

**[Online Mode]** Install required packages (better for development):
```bash
git clone https://github.com/ZhixuanWuA/MViQA-DOTS
cd MViQA-DOTS
pip install -r requirements.txt
```

**[Offline Mode]** Install VideoLLaMA2 as a Python package (better for direct use):
```bash
git clone https://github.com/ZhixuanWuA/MViQA-DOTS
cd MViQA-DOTS
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## 📊 Datasets
We conduct experiments on [MSVD-QA](https://github.com/xudejing/video-question-answering) , [MSRVTT-QA](https://github.com/xudejing/video-question-answering), and [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) datasets. 
Then extract video frames from each video at 10 fps (https://github.com/boheumd/MA-LMM/blob/main/data/extract_frames.py), based on the annotations of each dataset, object detection with pre-trained RAM++ (https://github.com/IDEA-Research/Grounded-Segment-Anything).
Suppose your data structure is like:
```bash
datasets
│   ├── MSVD
│   |   ├── frames
│   |   ├── videos
|   |   └── annoations
|   |   └── objectdetected
...
```


## 🗝️ Training & Evaluation
**[Pre-trained LLM]** 
We use the pre-trained Q-Former weights from [InstructBLIP](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) and [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) as the LLM.

**[Training]** 
