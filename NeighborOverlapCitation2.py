import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial

from ogb.linkproppred import Evaluator
from ogbdataset import loaddataset
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = []
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    for perm in PermIterator(
            source_edge.device, source_edge.shape[0], batch_size
    ): 
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0)
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   source_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        
        src, dst = source_edge[perm], target_edge[perm]
        pos_out = predictor(h, adj, torch.stack((src, dst)))

        pos_loss = -F.logsigmoid(pos_out).mean()

        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h, adj, torch.stack((src, dst_neg)))
        neg_loss = -F.logsigmoid(-neg_out).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()
    adj = data.full_adj_t
    h = model(data.x, adj)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h, adj, torch.stack((src, dst))).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h, adj, torch.stack((src, dst_neg))).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = 0.0 #test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr, h.cpu()


def parseargs():
    # GCN params {'hiddim': 128, 'mplayers': 1, 'lr': 0.001, 'batch_size': 65536, 'maskinput': True, 'dropout': 0.2, 'res': True}
    # INCN1 params {'hiddim': 32, 'mplayers': 1, 'lr': 0.0003, 'batch_size': 8192, 'maskinput': True, 'dropout': 0.3, 'res': True, 'model': 'max', 'scale': 6.4, 'offset': 0.95}
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--use_valedges_as_input", action="store_true")
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    data, split_edge = loaddataset(args.dataset, False, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor in ["incn1cn1", "sincn1cn1"]:
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []

    for run in range(args.runs):
        set_seed(run)
        bestscore = [0, 0, 0]
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp).to(device)

        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])

        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput)
            print(f"trn time {time.time()-t1:.2f} s")
            if True:
                t1 = time.time()
                results = test(model, predictor, data, split_edge, evaluator,
                               args.testbs)
                results, h = results[:-1], results[-1]
                print(f"test time {time.time()-t1:.2f} s")
                writer.add_scalars(f"mrr_{run}", {
                        "trn": results[0],
                        "val": results[1],
                        "tst": results[2]
                    }, epoch)

                if True:
                    train_mrr, valid_mrr, test_mrr = results
                    train_mrr, valid_mrr, test_mrr = results
                    if valid_mrr > bestscore[1]:
                        bestscore = list(results) 
                        bestscore = list(results) 
                        if args.save_gemb:
                            torch.save(h, f"gemb/citation2_{args.model}_{args.predictor}.pt")

                    print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_mrr:.2f}%, '
                              f'Valid: {100 * valid_mrr:.2f}%, '
                              f'Test: {100 * test_mrr:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "citation2":
            ret.append(bestscore)
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: {np.average(ret[:, 1])} {np.std(ret[:, 1])} {np.average(ret[:, 2])} {np.std(ret[:, 2])}")

if __name__ == "__main__":
    main()