import optuna

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--dev", type=int)
args = parser.parse_args()

stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.{args.model}.db", study_name=f"{args.model}", load_if_exists=True, direction="maximize")

def obj(trial: optuna.Trial, dev: int=args.dev, dataset="ddi"):
    runs = # fixed. Larger runs leads to more stable score for each trial. You can increase it during tuning.
    batch_size =  # fixed by GPU memory, as large as possible 
    hiddim = # fixed
    testbs = # fixed by GPU memory, as large as possible
    epochs =  # fixed  
    maskinput = trial.suggest_categorical("maskinput", [True, False]) 
    jk = trial.suggest_categorical("jk", [True, False]) 
    res = trial.suggest_categorical("res", [True, False]) 
    use_xlin = trial.suggest_categorical("use_xlin", [True, False]) 
    tailact = trial.suggest_categorical("tailact", [True, False]) 
    twolayerlin = trial.suggest_categorical("twolayerlin", [True, False]) 
    model = trial.suggest_categorical("model", [True, False]) 
    xdp = trial.suggest_float("xdp", 0.0, 0.9, step=0.05) 
    tdp = trial.suggest_float("tdp", 0.0, 0.9, step=0.05) 
    gnnedp = trial.suggest_float("gnnedp", 0.0, 0.9, step=0.05)
    preedp = trial.suggest_float("preedp", 0.0, 0.9, step=0.05)
    gnndp = trial.suggest_float("gnndp", 0.0, 0.9, step=0.05) 
    predp = trial.suggest_float("predp", 0.0, 0.9, step=0.05) 
    gnnlr = trial.suggest_float("gnnlr", 1e-4, 1e-2, step=3e-4) 
    prelr = trial.suggest_float("prelr", 1e-4, 1e-2, step=3e-4) 
    mplayers = trial.suggest_int("mplayers", 1, 3) 
    alpha = trial.suggest_float("alpha", 0.3, 1.0, step=0.1) 
    model = trial.suggest_categorical("model", ["gcn", "sage", "gin", "max", "puremax", "puremean", "puresum", "puregcn"]) 
    cmd = f"CUDA_VISIBLE_DEVICES={dev} nohup python NeighborOverlap.py " 
    cmd +=f" --xdp {xdp} --tdp {tdp} --gnnedp {gnnedp} --preedp {preedp} --predp {predp} --gnndp {gnndp} --gnnlr {gnnlr} --prelr {prelr}  --batch_size {batch_size} "
    cmd +=f" --ln --lnnn --predictor cn1 --dataset {dataset} "
    cmd +=f" --epochs {epochs} --runs {runs} --model {model} --alpha {alpha} --hiddim {hiddim} --mplayers {mplayers}  --testbs {testbs}  "
    if use_xlin:
        cmd += " --use_xlin " 
    if tailact:
        cmd += " --tailact " 
    if twolayerlin:
        cmd += " --twolayerlin " 
    if res:
        cmd += " --res "
    if jk:
        cmd += " --jk " 
    if maskinput:
        cmd += " --maskinput " 
    cmd += f"|grep Final"

    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-5]) - float(ret.split()[-4])
    return out
   

def objncnc(trial: optuna.Trial, dev: int=args.dev, dataset="ddi"):
    preedp = trial.suggest_float("preedp", 0.0, 0.9, step=0.05)
    predp = trial.suggest_float("predp", 0.0, 0.9, step=0.05) 
    scale = trial.suggest_float("scale",1, 20, step=0.5)
    offset = trial.suggest_float("offset", 3, 20, step=0.5)
    pt = trial.suggest_float("pt", 0.1, 0.9,  step=0.1)
    alpha = trial.suggest_float("alpha", 0.3, 1, step=0.1) 
    prelr = trial.suggest_float("prelr", 1e-4, 1e-2, step=3e-4) 
    ''' 
    Please change the parameter in the cmd to that of NCN
    ''' 
    cmd = f"CUDA_VISIBLE_DEVICES={dev}  nohup python NeighborOverlap.py " 
    cmd +=f"  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp {preedp} --predp {predp} --gnndp 0.4 --gnnlr 0.0021 --prelr {prelr} --batch_size 24576 "
    cmd +=f" --ln --lnnn --predictor incn1cn1 --dataset {dataset} --proboffset {offset} --probscale {scale} --pt {pt} --alpha {alpha} "
    cmd +=f" --epochs 10 --runs 1 --model puresum --hiddim 224 --mplayers 1  --testbs 24576 --splitsize 262144    --use_xlin  --twolayerlin  --res  --maskinput --loadmod  " 
    cmd += f"|grep Final"
    ret = subprocess.check_output(cmd, shell=True)
    ret = str(ret, encoding="utf-8")
    print(cmd, flush=True)
    print(ret, flush=True)
    out = float(ret.split()[-5]) - float(ret.split()[-4])
    return out

if args.model == "incn1cn1":
    stu.optimize(objncnc, 100)
elif args.model == "cn1":
    stu.optimize(obj, 100)