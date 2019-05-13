import os
import numpy as np
import pandas as pd
import sys
import argparse
import glob

"""
Gridsearch script for citation datasets

Available models: 
  'gcn_ae': autoencoder model, 
  'gcn_vae': variational autoencoder, 
  'gcn_hvae': poincare wasserstein autoencoder (pwa), 
  'gcn_hvae_ff': pwa with feedforward hyperbolic layers on top of gcn layers

Sample_expmap flag switches between rejection based sampling of the prior
and the exponential map projection of the Euclidean N(0,1) samples 
"""


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--folder', type=str, help='Data folder')
  parser.add_argument('--dataset', type=str, help='Dataset')
  parser.add_argument('--model', type=str, help='Model')
  parser.add_argument('--seed', default=42, type=int, help='Model')
  parser.add_argument('--sample_expmap',default=False, action="store_true",
                      help='prior sample via exponential map')
  args = parser.parse_args()

  lrs = [0.001, 0.005, 0.01]
  # lrs = [0.1]
  dropout = [0.0, 0.1, 0.2, 0.4]
  z_dim = [8,16,32]
  mmd_beta = [10,50,200]
  dataset = args.dataset
  model = args.model
  seed = args.seed
  print(seed)
  if args.sample_expmap:
    em_flag = "em1"
  else:
    em_flag = "em0"

  for lr in lrs:
    for dr in dropout:
      for dim in z_dim:
        for beta in mmd_beta:
          cmd = "python train.py --mode train --model {} --dataset {}\
              --lr_adam {} --mmd_beta {} --hidden2 {} --seed {}\
              --dropout {}\
              --expname {}".format(model,dataset,lr,beta,dim,seed,
                                   dr,dataset+model+em_flag+str(seed)+str(lr)+str(beta)+str(dim)+str(dr))
          if args.sample_expmap:
            cmd = cmd + " --sample_expmap"
          print(cmd)
          os.system(cmd)
          cmd = "python train.py --mode test --model gcn_hvae {} --dataset {}\
              --lr_adam {} --mmd_beta {} --hidden2 {} --seed {}\
              --dropout {}\
              --expname {}".format(model,dataset,lr,beta,dim,seed,
                                   dr,dataset+model+em_flag+str(seed)+str(lr)+str(beta)+str(dim)+str(dr))
          if args.sample_expmap:
            cmd = cmd + " --sample_expmap"
          print(cmd)
          os.system(cmd)

