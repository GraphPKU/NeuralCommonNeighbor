CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --gnnlr 0.0043 --prelr 0.0024 --batch_size 1152 --ln --lnnn --predictor pcn1 --dataset Cora --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > cora.pcn1.out &
CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --gnnlr 0.0043 --prelr 0.0024 --batch_size 1152 --ln --lnnn --predictor pincn1cn1 --dataset Cora --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > cora.pincn1cn1.out &
CUDA_VISIBLE_DEVICES=5 nohup python NeighborOverlap.py --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75 --probscale 6.5 --proboffset 4.4 --alpha 0.4 --gnnlr 0.0085 --prelr 0.0078 --batch_size 384 --ln --lnnn --predictor pcn1 --dataset Citeseer --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 4096 --maskinput --jk --use_xlin --tailact --twolayerlin > citeseer.pcn1.out &
CUDA_VISIBLE_DEVICES=5 nohup python NeighborOverlap.py --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75 --probscale 6.5 --proboffset 4.4 --alpha 0.4 --gnnlr 0.0085 --prelr 0.0078 --batch_size 384 --ln --lnnn --predictor pincn1cn1 --dataset Citeseer --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 4096 --maskinput --jk --use_xlin --tailact --twolayerlin > citeseer.pincn1cn1.out &
CUDA_VISIBLE_DEVICES=6 nohup python NeighborOverlap.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor pcn1 --dataset Pubmed --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > pubmed.pcn1.out &
CUDA_VISIBLE_DEVICES=7 nohup python NeighborOverlap.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor pincn1cn1 --dataset Pubmed --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1 --testbs 8192 --maskinput --jk --use_xlin --tailact > pubmed.pincn1cn1.out &
wait
CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013 --batch_size 16384 --ln --lnnn --predictor pcn1 --dataset ppa --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput --tailact --res --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 --savemod > ppa.pcn1.out &
CUDA_VISIBLE_DEVICES=6 nohup python NeighborOverlap.py --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1 --probscale 2.5 --proboffset 6.0 --alpha 1.05 --gnnlr 0.0082 --prelr 0.0037 --batch_size 65536 --ln --lnnn --predictor pcn1 --dataset collab --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1 --testbs 131072 --maskinput --use_valedges_as_input --res --use_xlin --tailact > collab.pcn1.out &
CUDA_VISIBLE_DEVICES=7 nohup python NeighborOverlap.py --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1 --probscale 2.5 --proboffset 6.0 --alpha 1.05 --gnnlr 0.0082 --prelr 0.0037 --batch_size 65536 --ln --lnnn --predictor pincn1cn1 --dataset collab --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1 --testbs 131072 --maskinput --use_valedges_as_input --res --use_xlin --tailact > collab.pincn1cn1.out &
wait
CUDA_VISIBLE_DEVICES=1 nohup python NeighborOverlap.py --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0 --predp 0.6 --gnndp 0.4 --gnnlr 0.0021 --prelr 0.0018 --batch_size 24576 --ln --lnnn --predictor pcn1 --dataset ddi --epochs 100 --runs 10 --model puresum --hiddim 224 --mplayers 1 --testbs 131072 --use_xlin --twolayerlin --res --maskinput --savemod > ddi.pcn1.out &
CUDA_VISIBLE_DEVICES=5 nohup python NeighborOverlap.py --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013 --batch_size 16384 --ln --lnnn --predictor pincn1cn1 --dataset ppa --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput --tailact --res --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 --savemod > ppa.pincn1cn1.out &