import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import DatasetConfig
from model.mixed_regressor import MixedRegressor
from util import create_dir, create_logger, load_csv, save_csv


def bresenham_line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points


def is_path_clear(map, start, end):
    x1, y1 = start
    x2, y2 = end

    points = bresenham_line(x1, y1, x2, y2)

    for x, y in points:
        if map[x][y] == 1:
            return 0
    return 1


def check_paths(map, start, ob_pos):
    results = []
    for pos in ob_pos:
        results.append(is_path_clear(map, start, pos))

    return np.array(results)


def process_data(config, floor_map, pos, ob_pos, rss):
    dist = np.empty((len(pos), len(ob_pos)))
    for index, ob in enumerate(ob_pos):
        dist[:, index:index + 1] = np.sqrt(((pos - ob) ** 2).sum(axis=1)).reshape(-1, 1)

    con = np.zeros((len(pos), len(ob_pos)))
    for i, p in enumerate(pos):
        con[i] = check_paths(floor_map, p, ob_pos)

    dist_tensor = torch.FloatTensor(dist)
    pos_tensor = torch.FloatTensor(pos)
    con_tensor = torch.FloatTensor(con)

    rss_tensor = torch.FloatTensor(rss / config['min_rss'])

    return dist_tensor, pos_tensor, con_tensor, rss_tensor


class FPDataset(Dataset):
    def __init__(self, config):
        floor_map_data = load_csv(config['floor_plan_path'], header=False).values[:, 2]
        self.floor_map = floor_map_data.reshape(config['m'], config['n'])

        ob_data = load_csv(config['ob_data_path'], header=False).values
        self.ob_pos = np.unique(ob_data[:, :2], axis=0)
        config['ob_size'] = len(self.ob_pos)

        self.t_data = load_csv(config['labeled_data_path'], header=False).values
        self.t_pos, self.t_rss = self.t_data[:, :2], self.t_data[:, 2:]
        self.t_dist_tensor, self.t_pos_tensor, self.t_con_tensor, self.t_rss_tensor = process_data(config, self.floor_map, self.t_pos, self.ob_pos, self.t_rss)

        self.v_data = load_csv(config['v_data_path'], header=True).values
        self.v_pos, self.v_rss = self.v_data[:, :2], self.v_data[:, 2:]
        self.v_dist_tensor, self.v_pos_tensor, self.v_con_tensor, _ = process_data(config, self.floor_map, self.v_pos, self.ob_pos, self.v_rss)

    def __len__(self):
        return len(self.t_rss_tensor)

    def __getitem__(self, idx):
        return self.t_dist_tensor[idx], self.t_pos_tensor[idx], self.t_con_tensor[idx], self.t_rss_tensor[idx]


def evaluate(config, model, dataset):
    v_dist_tensor = dataset.v_dist_tensor.to(config['device'])
    v_pos_tensor = dataset.v_pos_tensor.to(config['device'])
    v_con_tensor = dataset.v_con_tensor.to(config['device'])

    model.eval()
    with torch.no_grad():
        rss_bar = model(v_dist_tensor, v_pos_tensor, v_con_tensor)
        rss_bar = rss_bar.detach().cpu().numpy() * config['min_rss']

    fp_with_pos = np.concatenate((dataset.v_pos, rss_bar), axis=1)
    fp_map = pd.DataFrame(fp_with_pos).groupby([0, 1]).mean().reset_index().values

    return fp_map


def train_model(config, logger, dataset):
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)
    model = MixedRegressor(dist_input_dim=config['ob_size'], pos_input_dim=2, con_input_dim=config['ob_size'],
                            dist_hidden_dim=128, pos_hidden_dim=32, con_hidden_dim=128,combined_hidden_dim=64,
                            output_dim=config['ap_size']
                           ).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    res_dict = {
        "fp_map": None,
    }

    cur_patience = 0
    best_loss = sys.float_info.max

    for epoch in range(config['epoch']):
        model.train()
        total_loss = 0
        for t_dist, t_pos, t_con, t_rss in dataloader:
            t_dist = t_dist.to(config['device'])
            t_pos = t_pos.to(config['device'])
            t_con = t_con.to(config['device'])
            t_rss = t_rss.to(config['device'])

            rss_bar = model(t_dist, t_pos, t_con)
            loss = F.mse_loss(rss_bar * config['min_rss'], t_rss * config['min_rss'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        fp_map = evaluate(config, model, dataset)

        res_dict["fp_map"] = fp_map

        logger.info(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            cur_patience = 0
        else:
            cur_patience += 1
            if cur_patience == config['patience']:
                break

    return res_dict


def exp(config):
    config["labeled_data_path"] = f"./dataset/{config['name']}/sample/p{config['percent']}_l{config['layout']}.csv"
    config["ob_data_path"] = f"./dataset/{config['name']}/sample/ob_p{config['percent']}_l{config['layout']}.csv"
    fp_map_dir_path = create_dir(f"dataset/{config['name']}/fp_map/step1/p{config['percent']}_l{config['layout']}")
    logger = create_logger(f"{fp_map_dir_path}/out.log")

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['patience'] = 10

    logger.info(config)

    dataset = FPDataset(config)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    res_dict = train_model(config, logger, dataset)

    save_csv(res_dict['fp_map'], f"{fp_map_dir_path}/step1.csv")


def run_ssb():
    config = DatasetConfig.ssb

    config['epoch'] = 100
    config['batch_size'] = 50
    config['learning_rate'] = 0.005
    config['patience'] = 10

    config['percent'], config['layout'] = 30, 0
    exp(config)


if __name__ == "__main__":
    run_ssb()

