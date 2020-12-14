# Spatio-attentive Graphs

__We are currently cleaning up the repository...__

---
Official PyTorch implementation for our paper [Spatio-attentive Graphs for Human-Object Interaction Detection](https://arxiv.org/pdf/2012.06060.pdf)

<img src="./assets/bipartite_graph.png" alt="bipartite_graph" height="240" align="left"/>
<img src="./assets/zoom_in.png" alt="zoom_in" height="240" align="left"/>
<img src="./assets/mutibranch_fusion.png" alt="mutibranch_fusion" height="240" align="center"/>

## Citation

If you find this repository useful for your research, please kindly cite our paper:

```bibtex
@article{zhang2020,
	author = {Frederic Z. Zhang, Dylan Campbell and Stephen Gould},
	title = {Spatio-attentive Graphs for Human-Object Interaction Detection},
	journal = {arXiv preprint arXiv:2012.06060},
	year = {2020}
}
```

## Prerequisites

1. Download the repository with `git clone https://github.com/fredzzhang/spatio-attentive-graphs`
2. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
3. Make sure the environment you created for Pocket is activated. You are good to go!

## Data

1. Download the repository for [HICO-DET utilities](https://github.com/fredzzhang/hicodet)
2. Generate detections by running a pre-trained object detector following the [instructions](https://github.com/fredzzhang/hicodet/tree/main/detections#generate-detections-using-faster-r-cnn)
3. Create a softlink `ln -s /path/to/hicodet /path/to/spatio-attentive-graphs/hicodet`

## Training

Please wait for further instructions...

## Testing

Please wait for further instructions...