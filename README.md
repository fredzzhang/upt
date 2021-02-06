# Spatio-attentive Graphs

Official PyTorch implementation for our paper [Spatio-attentive Graphs for Human-Object Interaction Detection](https://arxiv.org/pdf/2012.06060.pdf)

<img src="./assets/bipartite_graph.png" alt="bipartite_graph" height="200" align="left"/>
<img src="./assets/zoom_in.png" alt="zoom_in" height="200" align="left"/>
<img src="./assets/mutibranch_fusion.png" alt="mutibranch_fusion" height="200" align="center"/>

## Citation

If you find this repository useful for your research, please kindly cite our paper:

```bibtex
@article{zhang2020,
	author = {Frederic Z. Zhang and Dylan Campbell and Stephen Gould},
	title = {Spatio-attentive Graphs for Human-Object Interaction Detection},
	journal = {arXiv preprint arXiv:2012.06060},
	year = {2020}
}
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Utilities](#data-utilities)
    * [HICO-DET](#hico-det)
    * [V-COCO](#v-coco)
- [Testing](#testing)
    * [HICO-DET](#hico-det-1)
    * [V-COCO](#v-coco-1)
- [Training](#training)
    * [HICO-DET](#hico-det-2)
    * [V-COCO](#v-coco-2)
- [Contact](#contact)

## Prerequisites

1. Download the repository with `git clone https://github.com/fredzzhang/spatio-attentive-graphs`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!

## Data Utilities

The [HICO-DET](https://github.com/fredzzhang/hicodet) and [V-COCO](https://github.com/fredzzhang/vcoco) repos have been incorporated as submodules for convenience. To download relevant data utilities, run the following commands.
```bash
cd /path/to/spatio-attentive-graphs
git submodule init
git submodule update
```
### HICO-DET
1. Download the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/spatio-attentive-graphs/hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/spatio-attentive-graphs/hicodet
    ln -s /path/to/hico_20160224_det ./hico_20160224_det
    ```
2. Run a Faster R-CNN pre-trained on MS COCO to generate detections
```bash
cd /path/to/spatio-attentive-graphs/hicodet/detections
python preprocessing.py --partition train2015
python preprocessing.py --partition test2015
```
3. Generate ground truth detections (optional)
```bash
cd /path/to/spatio-attentive-graphs/hicodet/detections
python generate_gt_detections.py --partition test2015 
```
4. Download fine-tuned detections (optional)
```bash
cd /path/to/spatio-attentive-graphs/download
bash download_finetuned_detections.sh
```
To attempt fine-tuning yourself, refer to the [instructions](https://github.com/fredzzhang/hicodet/tree/main/detections#fine-tune-the-detector-on-hico-det) in the [HICO-DET repository](https://github.com/fredzzhang/hicodet). The checkpoint of our fine-tuned detector can be found [here](https://drive.google.com/file/d/11lS2BQ_In-22Q-SRTRjRQaSLg9nSim9h/view?usp=sharing).

### V-COCO
1. Download the `train2014` and `val2014` partitions of the [COCO dataset](https://cocodataset.org/#download)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/spatio-attentive-graphs/vcoco
    bash download.sh
    ```
    2. If you have previsouly downloaded the dataset, simply create a soft link. Note that 
    ```bash
    cd /path/to/spatio-attentive-graphs/vcoco
    ln -s /path/to/coco ./mscoco2014
    ```
2. Run a Faster R-CNN pre-trained on MS COCO to generate detections
```bash
cd /path/to/spatio-attentive-graphs/vcoco/detections
python preprocessing.py --partition trainval
python preprocessing.py --partition test
```
## Testing
### HICO-DET
1. Download the checkpoint of our trained model
```bash
cd /path/to/spatio-attentive-graphs/download
bash download_checkpoint.sh
```
2. Test a model
```bash
cd /path/to/spatio-attentive-graphs
CUDA_VISIBLE_DEVICES=0 python test.py --model-path checkpoints/weights-hicodet-b32h16e11.pt
```
By default, detections from a pre-trained detector is used. To change sources of detections, use the argument `--detection-dir`, e.g. `--detection-dir hicodet/detections/test2015_gt` to select ground truth detections. Fine-tuned detections (if you downloaded them) are available under `hicodet/detections`.

3. Cache detections for Matlab evaluation following [HO-RCNN](https://github.com/ywchao/ho-rcnn) (optional)
```bash
cd /path/to/spatio-attentive-graphs
CUDA_VISIBLE_DEVICES=0 python cache.py --model-path checkpoints/weights-hicodet-b32h16e11.pt
```
By default, 80 `.mat` files, one for each object class, will be cached in a directory named `matlab`. Use the `--cache-dir` argument to change the cache directory. To change sources of detections, refer to the use of `--detection-dir` in the previous section.

As a reference, the performance of the provided model is shown in the table below

|Detections|Default Setting|Known Object Setting|
|:-|:-:|:-:|
|Pre-trained on MS COCO|(`19.39`, `15.05`, `20.68`)|(`22.71`, `17.72`, `24.20`)|
|Fine-tuned on HICO-DET ([VCL](https://drive.google.com/file/d/1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4/view))|(`22.61`, `16.30`, `24.49`)|(`25.11`, `18.05`, `27.22`)|
|Fine-tuned on HICO-DET ([This repo](https://github.com/fredzzhang/hicodet))|(`23.57`, `16.93`, `25.55`)|(`26.48`, `18.79`, `28.77`)|
|Fine-tuned on HICO-DET ([DRG](https://drive.google.com/file/d/18_6K2P6s9vMBWOvcNNQqUj2wfLhbvpLo/view))|(`29.55`, `22.61`, `31.62`)|(`31.92`, `24.24`, `34.21`)|
|Ground truth detections|(`49.26`, `35.82`, `53.27`)|(`49.40`, `35.85`, `53.45`)|

### V-COCO

We did not implement evaluation utilities for V-COCO, and instead use the [utilities](https://github.com/s-gupta/v-coco#evaluation) provided by Gupta. To generate the required pickle file, run the following script by correctly specifying the path to a model with `--model-path`

```bash
cd /path/to/spatio-attentive-graphs
CUDA_VISIBLE_DEVICES=0 python cache.py --dataset vcoco --data-root vcoco \
    --detection-dir vcoco/detections/test \
    --cache-dir vcoco_cache --partition test \
    --model-path /path/to/a/model
```

This will generate a file named `vcoco_results.pkl` under `vcoco_cache` in the current directory. Please refer to the [v-coco](https://github.com/s-gupta/v-coco) repo (not to be confused with [vcoco](https://github.com/fredzzhang/vcoco), the submodule) for further instructions. __Note__ that loading the pickle file requires a particular class `CacheTemplate`, which is shown below in its entirety.
```python
from collections import defaultdict
class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]
```
You can either add it into the evaluation code or save it as a seperate file to import from.

## Training
### HICO-DET
1. Training with `DistributedDataParallel` (_recommended_)
```bash
cd /path/to/spatio-attentive-graphs
python main_dist.py --world-size 8 --learning-rate 0.008 --print-interval 300 &>log &
```
Specify the number of GPUs to use with the argument `--world-size`. The default batch size is `4` (per GPU). The provided model was trained with 8 GPUs, with an effective batch size of `32`. __Reducing the effective batch size could result in slightly inferior performance__. The default learning rate for batch size of 4 is `0.001`. As a rule of thumb, multiply that by the same factor by which batch size has been increased, e.g. `0.008` for batch size of `32`. It is recommended to redirect `stdout` and `stderr` to a file to save the training log (as indicated by `&>log`). To check the progress, run `cat log | grep mAP`, or alternatively you can go through the log with `vim log`. Also, the mAP logged follows a slightly different protocol. It does __NOT__ necessarily correlate with the mAP that the community reports. It only serves as a diagnostic tool. The true performance of the model requires running a seperate test as shown in the previous section. By default, checkpoints will be saved under `checkpoints` in the current directory. For more arguments, run `python main_dist.py --help` to find out. We follow the early stopping training strategy, and have concluded (using a validation set split from the training set) that the model at epoch `11` should be picked.

2. Training with a single GPU (_inferior performance_)
```bash
cd /path/to/spatio-attentive-graphs
CUDA_VISIBLE_DEVICES=0 python main.py &>log &
```
Note that setting the environmental variable `CUDA_VISIBLE_DEVICES` is necessary and should __NOT__ be omitted, due to the design of the used learning engine. The training process takes about 40 hrs on a single GeForce GTX TITAN X device. Therefore it is recommended to run the script asynchronously (as indicated by `&`). For more arguments, run `python main.py --help` to find out.

### V-COCO
We have incorporated V-COCO dataset into the training pipeline. Only a few arguments need to be modified to train on V-COCO.

```bash
cd /path/to/spatio-attentive-graphs
# Training with DDP (recommended)
python main_dist.py --dataset vcoco --partitions trainval val --data-root vcoco \
    --train-detection-dir vcoco/detections/trainval \
    --val-detection-dir vcoco/detections/trainval \
    --learning-rate 0.008 --world-size 8 \
    --print-interval 20 --cache-dir checkpoints/vcoco &>log &
```

## Contact

If you have any questions regarding our paper or the repo, please post them in [discussions](https://github.com/fredzzhang/spatio-attentive-graphs/discussions). If you ran into issues related to the code, feel free to open an issue. Alternatively, you can contact me at frederic.zhang@anu.edu.au