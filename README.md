# MViQA-DOTS
Multi-modal video question-answering
<p align="center">
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
