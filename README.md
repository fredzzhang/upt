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

## Prerequisites

1. Download the repository with `git clone https://github.com/fredzzhang/spatio-attentive-graphs`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!

## Data Utilities

The [HICO-DET repository](https://github.com/fredzzhang/hicodet) has been incorporated as a submodule for convenience.
1. Download data utilities
```bash
cd /path/to/spatio-attentive-graphs
git submodule init
git submodule update
```
2. Prepare the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd hicodet
    ln -s /path/to/hico_20160224_det ./hico_20160224_det
    ```
3. Run a Faster R-CNN pre-trained on MS COCO to generate detections
```bash
cd detections
python preprocessing.py --partition train2015
python preprocessing.py --partition test2015
```
4. Generate ground truth detections (optional)
```bash
python generate_gt_detections.py --partition test2015 
```
5. Download fine-tuned detections (optional)
```bash
cd ../../download
bash download_finetuned_detections.sh
```

## Training

Please wait for further instructions...

## Testing

Please wait for further instructions...
