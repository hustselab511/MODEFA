import logging
import os
import time

import numpy as np
import pandas as pd
from seaborn.external.docscrape import header
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.vector_ar.svar_model import svar_ckerr


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def create_logger(file_path=None):
    logger = logging.getLogger(str(time.time()))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if file_path is not None:
        fh = logging.FileHandler(file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_csv(data, file_path, header=False):
    if header:
        pd.DataFrame(data).to_csv(file_path, index=False, header=True)
    else:
        pd.DataFrame(data).to_csv(file_path, index=False, header=False)


def load_csv(file_path, header=False):
    if header:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, header=None)


def cal_metric(y_true, y_pred):
    distance_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    mae = np.mean(distance_errors)
    rmse = np.sqrt(np.mean(distance_errors ** 2))

    return mae, rmse


def reshape_fp_map(config, fp_map):
    fp_map_reshaped = np.zeros((config['ap_size'], config['n'], config['m']))

    for i in range(config['ap_size']):
        rss = fp_map[:, i]
        rss_reshaped = rss.reshape(config['m'], config['n'])
        fp_map_reshaped[i] = rss_reshaped.T

    return fp_map_reshaped


def inverse_reshape_fp_map(config, fp_map_reshaped):
    fp_map = np.zeros((config['m'] * config['n'], config['ap_size']))

    for i in range(config['ap_size']):
        rss = fp_map_reshaped[i]
        _rss = rss.T
        fp_map[:, i] = _rss.flatten()

    return fp_map

