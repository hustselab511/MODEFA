import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import DatasetConfig
from model.mixed_at_loc import MixedATLoc
from model.tf_ae import TFAutoencoder, TFAutoencoderMode
from util import create_logger, load_csv, cal_metric


def get_multi_scale_feat(config, fp_map_rss, patch_settings, compression_model, ob_rss):
    patch_feats_list = []

    fp_map_rss_2d = fp_map_rss.reshape((config['m'], config['n'], config['ap_size']))
    width, height, ap_size = fp_map_rss_2d.shape

    for patch_size, stride in patch_settings:
        patch_rss = []
        patch_w, patch_h = patch_size
        stride_w, stride_h = stride
        num_patches_w = (width - patch_w) // stride_w + 1
        num_patches_h = (height - patch_h) // stride_h + 1

        for i in range(num_patches_w):
            for j in range(num_patches_h):
                start_w = i * stride_w
                start_h = j * stride_h
                patch = fp_map_rss_2d[start_w:start_w + patch_w, start_h:start_h + patch_h, :]

                patch_mean = patch.mean(axis=(0, 1))

                patch_rss.append(patch_mean)

        patch_rss = np.array(patch_rss)
        patch_feat_tensor = get_diff_feat_tensor(config, compression_model, patch_rss, ob_rss)

        patch_feats_list.append(patch_feat_tensor)

    return patch_feats_list


def get_diff_feat_tensor(config, compression_model, rss, ob_rss):
    rss_size, ob_size, ap_size = rss.shape[0], ob_rss.shape[0], ob_rss.shape[1]
    diff_rss = np.zeros((rss_size, ob_size, ap_size))

    for i in range(rss_size):
        for j in range(ob_size):
            diff_rss[i, j] = rss[i] - ob_rss[j]

    compression_model.eval()
    with torch.no_grad():
        diff_rss_tensor = torch.FloatTensor(diff_rss / config['min_rss'])
        diff_feat_tensor = compression_model(diff_rss_tensor, TFAutoencoderMode.EVAL)

    return diff_feat_tensor


class FPDataset(Dataset):
    def __init__(self, config, compression_model):
        ob_data = load_csv(config['ob_data_path'], header=False)
        ob_data_mean = ob_data.groupby([0, 1]).mean().reset_index().values
        config['ob_size'] = len(ob_data_mean)
        self.ob_rss = ob_data_mean[:, 2:]

        fp_map = load_csv(config['fp_map_path'], header=False).values
        fp_map_pos, fp_map_rss = fp_map[:, :2], fp_map[:, 2:]

        scale_settings = [
            ((12, 7), (6, 4)),
            ((6, 6), (3, 3)),
            ((4, 3), (2, 2))
        ]
        self.patch_feats_list = get_multi_scale_feat(config, fp_map_rss, scale_settings, compression_model, self.ob_rss)

        self.pos_tensor = torch.FloatTensor(fp_map_pos)
        self.diff_feat_tensor = get_diff_feat_tensor(config, compression_model, fp_map_rss, self.ob_rss)

        v_data = load_csv(config['v_data_path'], header=True).values
        self.v_pos, v_rss = v_data[:, :2], v_data[:, 2:]
        self.v_diff_feat_tensor = get_diff_feat_tensor(config, compression_model, v_rss, self.ob_rss)

    def __len__(self):
        return len(self.diff_feat_tensor)

    def __getitem__(self, idx):
        return self.diff_feat_tensor[idx], self.pos_tensor[idx]


def evaluate(config, model, dataset):
    patch_feats_list = [patch_feats.to(config['device']) for patch_feats in dataset.patch_feats_list]
    v_diff_feat_tensor = dataset.v_diff_feat_tensor.to(config['device'])
    v_pos = dataset.v_pos

    model.eval()
    with torch.no_grad():
        pos_bar = model(v_diff_feat_tensor, patch_feats_list)
        pos_bar = pos_bar.detach().cpu().numpy()

    v_mae, v_rmse = cal_metric(v_pos, pos_bar)

    return v_mae, v_rmse


def train_model(config, logger, dataset):
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)
    model = MixedATLoc(channel_dim=8, ob_size=config['ob_size'], hidden_dim=64, gating_hidden_dim=64, output_dim=2).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    res_dict = {
        "epoch": -1,
        "mae": sys.float_info.max,
    }

    cur_patience = 0
    best_loss = sys.float_info.max

    patch_feats_list = [patch_feats.to(config['device']) for patch_feats in dataset.patch_feats_list]
    for epoch in range(config['epoch']):
        model.train()
        total_loss = 0
        for diff_feat, pos in dataloader:
            diff_feat = diff_feat.to(config['device'])
            pos = pos.to(config['device'])

            pos_bar = model(diff_feat, patch_feats_list)
            loss = F.mse_loss(pos_bar, pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mae, rmse = evaluate(config, model, dataset)

        if mae < res_dict["mae"]:
            res_dict["epoch"] = epoch + 1
            res_dict["mae"] = mae

        logger.info(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}, "
                    f"Loc MAE: {mae:.4f}, Loc RMSE: {rmse:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            cur_patience = 0
        else:
            cur_patience += 1
            if cur_patience == config['patience']:
                break


def exp(config):
    fp_map_dir_path = f"./dataset/{config['name']}/fp_map/step2/p{config['percent']}_l{config['layout']}"
    config['fp_map_path'] = f"{fp_map_dir_path}/step2.csv"
    config["ob_data_path"] = f"./dataset/{config['name']}/sample/ob_p{config['percent']}_l{config['layout']}.csv"
    compression_state_dict_path = f"./dataset/{config['name']}/compression_saved/p{config['percent']}_l{config['layout']}/model.pth"
    logger = create_logger()

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(config)

    compression_model = TFAutoencoder(input_dim=20, d_model=8, dim_feedforward=128, nhead=4, e_num_layers=4, d_num_layers=2)
    compression_model.load_state_dict(torch.load(compression_state_dict_path))
    dataset = FPDataset(config, compression_model)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_model(config, logger, dataset)

def run_ssb():
    config = DatasetConfig.ssb

    config['epoch'] = 100
    config['batch_size'] = 20
    config['learning_rate'] = 0.001
    config['patience'] = 10

    config['percent'], config['layout'] = 30, 0
    exp(config)

if __name__ == "__main__":
    run_ssb()

