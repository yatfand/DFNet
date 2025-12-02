
## DFNet
A novel dual-stream encoder-decoder network that integrates RGB images and depth information through multi-level feature extraction and fusion mechanisms, significantly enhancing detection accuracy and boundary clarity in complex underwater scenes.

### Requirements
- Python 3.9
- Pytorch 2.6.0
- CUDA 11.8
- opencv-python 
- timm
- numpy
- Pillow
- pandas
- matplotlib
- tqdm
- thop
- scikit-learn

### Datasets
[USOD](https://pan.baidu.com/s/1afb062AoGw53ShG2ezbJxw?pwd=rhhb) (code:rhhb)

[USOD10K](https://pan.baidu.com/s/1XuuNCrF0iXGyYS7q4NOqoA?pwd=2s3k) (code:2s3k)

### Pretrained Model
[pvt_v2_b2 Backbone](https://pan.baidu.com/s/1Gc-UeQHObu68Z1Y3BFArfw?pwd=uh9a) (code:uh9a)

[DFNet Pretrained Model](https://pan.baidu.com/s/1O-fS1fjZMthpo2fYkO3cDA?pwd=6gkw) (code:6gkw)

### Saliency maps
SaliencyMaps/USOD10K

SaliencyMaps/USOD

### Training
python train.py
### Inference
python test.py
### Evaluation
python eval.py

