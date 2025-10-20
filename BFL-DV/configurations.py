import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--iid', default=True)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--global_epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'])
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', default=True)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda:0', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--Threshold', type=float, default=-3)
    parser.add_argument('--seed', type=float, default=42)
    parser.add_argument('--num_users', type=int, default=30)
    parser.add_argument('--num_clusters', type=int, default=3)
    parser.add_argument('--m_num', type=int, default=4)
    parser.add_argument('--thresholdMultiplier', type=float, default=1.2)
    parser.add_argument('--atk_mode', type=dict, default={'atk': 2, 'def1': 1, 'def2': 1},
                        help="atk1模型篡改 atk2恶意节点勾结 atk3恶意簇（不平衡情况下） def1簇内识别 def2簇间识别")
    parser.add_argument('--atk_node', type=list, default=[0, 1, 2, 3,4, 10, 11, 12, 13,14, 20, 21, 22, 23,24],
                        choices=[
                            [0, 10, 20],10,
                            [0, 1, 10, 11, 20,21],20,
                            [0, 1, 2, 10, 11, 12, 20, 21, 22],30,
                            [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],40,
                            [0, 1, 2, 3,4, 10, 11, 12, 13,14, 20, 21, 22, 23,24],50,
                            [0,1,2,3,4,5,10,20,21]

                        ])
    parser.add_argument('--FL_TYPE', type=str, default='Cluster',
                        choices=['FDL', 'Cluster'])
    parser.add_argument('--Enhancement', type=str, default='ADASYN',
                        choices=['ADASYN', 'SMOTE'])
    args = parser.parse_args()
    return args

