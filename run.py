import argparse
import os
from trainer.base import trainer

# from src.PDE_category.base import linear
from src.PDE_category.Burgers import Burgers1D, Burgers2D
from src.PDE_category.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.PDE_category.Heat import Heat2D_ComplexGeometry, Heat2D_LongTime, Heat2D_Multiscale, Heat2D_VaryingCoef, HeatND
from src.PDE_category.Helmholtz import Helmholtz2D
from src.PDE_category.Inverse import HeatInv, PoissonInv
from src.PDE_category.NS import NS2D_BackStep, NS2D_Classic, NS2D_LidDriven, NS2D_LongTime
from src.PDE_category.Poisson import Poisson1D, Poisson2D_Classic, Poisson2D_ManyArea, Poisson3D_ComplexGeometry, PoissonBoltzmann2D, PoissonND
from src.PDE_category.Wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime

pde_list = [Burgers1D] + \
    [GrayScottEquation, KuramotoSivashinskyEquation] + \
    [Heat2D_ComplexGeometry, Heat2D_LongTime, Heat2D_Multiscale, Heat2D_VaryingCoef, HeatND] + \
    [Helmholtz2D] + \
    [HeatInv, PoissonInv] + \
    [NS2D_BackStep, NS2D_Classic, NS2D_LidDriven, NS2D_LongTime] +\
    [Poisson1D, Poisson2D_Classic, Poisson2D_ManyArea, Poisson3D_ComplexGeometry, PoissonBoltzmann2D, PoissonND] + \
    [Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime]

EA_algo_list = ["AGA", "CMAES", "GA", "MOEAD", "NSGA2", "PSO", "xNES_Adam", "xNES_NAG"]

exp_num = 5 # num of experiment
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EAPINN runner')
    parser.add_argument('--device', type=str, default="4# 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-layers', type=str, default="16*4")
    parser.add_argument('--max_iters', type=int, default=5000)

    parser.add_argument('--algo', type=str, choices=EA_algo_list, default="NSGA2")
    parser.add_argument("--pop_size", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--init_stdev", type=float, default=0.02)

    command_args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = command_args.device
    for iter_num in range(1,exp_num+1):

        for pde in pde_list:

            
            train_task = pde.PDE(hidden_layers=command_args.hidden_layers)
            policy = pde.PINNsPolicy(train_task.net,
                                    train_task.num_params,
                                    train_task.format_params_fn,
                                    train_task.layout)

            trainer(pde=pde, policy=policy, train_task=train_task, config=command_args, iter_num=iter_num)






