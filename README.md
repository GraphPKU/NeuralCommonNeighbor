This repository contains the official code for the paper [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/pdf/2302.00890.pdf).

**Environment**

Tested Combination:
torch 1.13.0 + pyg 2.2.0 + ogb 1.3.5

```
conda env create -f env.yaml
```

**Prepare Datasets**

```
python ogbdataset.py
```

**Reproduce Results**

We implement the following models.

| name     | $model    | command change     |
|----------|-----------|--------------------|
| GAE      | cn0       |                    |
| NCN      | cn1       |                    |
| NCNC     | incn1cn1  |                    |
| NCNC2    | incn1cn1  | add --depth 2  --splitsize 131072    |
| GAE+CN   | scn1      |                    |
| NCN2     | cn1.5     |                    |
| NCN-diff | cn1res    |                    |
| NoTLR    | cn1       | delete --maskinput |

To reproduce the results, please modify the following commands as shown in the table above.

Cora
```
python NeighborOverlap.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $model --dataset Cora  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
```

Citeseer
```
python NeighborOverlap.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor $model --dataset Citeseer  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin
```

Pubmed
```
python NeighborOverlap.py   --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1  --probscale 5.3 --proboffset 0.5 --alpha 0.3  --gnnlr 0.0097 --prelr 0.002  --batch_size 2048  --ln --lnnn --predictor $model --dataset Pubmed  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
```

CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --gnnlr 0.0043 --prelr 0.0024 --batch_size 1152 --ln --lnnn --predictor cn1 --dataset Cora --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > cora.cn1.out &
CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --gnnlr 0.0043 --prelr 0.0024 --batch_size 1152 --ln --lnnn --predictor incn1cn1 --dataset Cora --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > cora.incn1cn1.out &
CUDA_VISIBLE_DEVICES=5 nohup python NeighborOverlap.py --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75 --probscale 6.5 --proboffset 4.4 --alpha 0.4 --gnnlr 0.0085 --prelr 0.0078 --batch_size 384 --ln --lnnn --predictor cn1 --dataset Citeseer --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 4096 --maskinput --jk --use_xlin --tailact --twolayerlin > citeseer.cn1.out &
CUDA_VISIBLE_DEVICES=6 nohup python NeighborOverlap.py --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75 --probscale 6.5 --proboffset 4.4 --alpha 0.4 --gnnlr 0.0085 --prelr 0.0078 --batch_size 384 --ln --lnnn --predictor incn1cn1 --dataset Citeseer --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 4096 --maskinput --jk --use_xlin --tailact --twolayerlin > citeseer.incn1cn1.out &
CUDA_VISIBLE_DEVICES=7 nohup python NeighborOverlap.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor cn1 --dataset Pubmed --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > pubmed.cn1.out &
CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor incn1cn1 --dataset Pubmed --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > pubmed.incn1cn1.out &


CUDA_VISIBLE_DEVICES=2 nohup python NeighborOverlap.py --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013 --batch_size 16384 --ln --lnnn --predictor cn1 --dataset ppa --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput --tailact --res --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 --savemod > ppa.cn1.out &
CUDA_VISIBLE_DEVICES=3 nohup python NeighborOverlap.py --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013 --batch_size 16384 --ln --lnnn --predictor incn1cn1 --dataset ppa --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput --tailact --res --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 --savemod > ppa.incn1cn1.out &
CUDA_VISIBLE_DEVICES=4 nohup python NeighborOverlap.py --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1 --probscale 2.5 --proboffset 6.0 --alpha 1.05 --gnnlr 0.0082 --prelr 0.0037 --batch_size 65536 --ln --lnnn --predictor cn1 --dataset collab --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1 --testbs 131072 --maskinput --use_valedges_as_input --res --use_xlin --tailact > collab.cn1.out &
CUDA_VISIBLE_DEVICES=5 nohup python NeighborOverlap.py --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1 --probscale 2.5 --proboffset 6.0 --alpha 1.05 --gnnlr 0.0082 --prelr 0.0037 --batch_size 65536 --ln --lnnn --predictor incn1cn1 --dataset collab --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1 --testbs 131072 --maskinput --use_valedges_as_input --res --use_xlin --tailact > collab.incn1cn1.out &
CUDA_VISIBLE_DEVICES=6 nohup python NeighborOverlap.py --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0021 --prelr 0.0018 --batch_size 24576 --ln --lnnn --predictor cn1 --dataset ddi --epochs 100 --runs 10 --model puresum --hiddim 224 --mplayers 1 --testbs 131072 --use_xlin --twolayerlin --res --maskinput --savemod > ddi.cn1.out &
collab
```
python NeighborOverlap.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor $model --dataset collab  --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
```

ppa
```
python NeighborOverlap.py  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013  --batch_size 16384  --ln --lnnn --predictor $model --dataset ppa   --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072
```

The following datasets use separate commands for NCN and NCNC. To use other models, please modify NCN's command. Note that NCNC models in these datasets initialize parameters with trained NCN models to accelerate training. Please use our pre-trained model or run NCN first.

citation2
```
python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.0058 --batch_size 32768 --ln --lnnn --predictor cn1 --dataset citation2 --epochs 20 --runs 10 --model puregcn --hiddim 64 --mplayers 3 --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 --save_gemb 


python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.001 --batch_size 24576 --ln --lnnn --predictor incn1cn1 --dataset citation2 --epochs 20 --runs 10 --model none --hiddim 64 --mplayers 0 --res --testbs 65536 --use_xlin --tailact --load gemb/citation2_puregcn_cn1.pt --proboffset -0.3 --probscale 1.4 --pt 0.25 --trndeg 96 --tstdeg 96 --load gemb/citation2_puregcn_cn1.pt 
```


 CUDA_VISIBLE_DEVICES=4 nohup python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.0058 --batch_size 32768 --ln --lnnn --predictor cn1 --dataset citation2 --epochs 20 --runs 10 --model puregcn --hiddim 64 --mplayers 3 --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 --save_gemb > citation2.out &
CUDA_VISIBLE_DEVICES=7 nohup python NeighborOverlapCitation2.py --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0 --predp 0.2 --gnndp 0.2 --gnnlr 0.0088 --prelr 0.0058 --batch_size 32768 --ln --lnnn --predictor pcn1 --dataset citation2 --epochs 20 --runs 10 --model puregcn --hiddim 64 --mplayers 3 --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 --save_gemb > citation2.pcn1.out &
CUDA_VISIBLE_DEVICES=6 nohup python NeighborOverlap.py --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013 --batch_size 16384 --ln --lnnn --predictor pincn1cn1 --dataset ppa --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput --tailact --res --testbs 32768 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 98304 --savemod > ppa.pincn1cn1.out

ddi
```
python NeighborOverlap.py  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0021 --prelr 0.0018  --batch_size 24576  --ln --lnnn --predictor cn1 --dataset ddi  --epochs 100 --runs 10 --model puresum --hiddim 224 --mplayers 1  --testbs 131072   --use_xlin  --twolayerlin  --res  --maskinput --savemod

python NeighborOverlap.py --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0000000 --prelr 0.0025 --batch_size 24576 --ln --lnnn --predictor incn1cn1 --dataset ddi --proboffset 3 --probscale 10 --pt 0.1 --alpha 0.5 --epochs 4 --runs 10 --model puresum --hiddim 224 --mplayers 1 --testbs 24576 --splitsize 262144 --use_xlin --twolayerlin --res --maskinput --loadmod
```