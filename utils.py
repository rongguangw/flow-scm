import os
import shutil
import numpy as np
import pandas as pd


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def milestone_step(args, optimizer, epoch):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
