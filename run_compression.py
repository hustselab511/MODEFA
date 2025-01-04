import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from config import DatasetConfig
from model.tf_ae import TFAutoencoder, TFAutoencoderMode
from util import load_csv, create_dir, create_logger


def process_cro_data(cro_rss, ob_rss):
    cro_size, ob_size, ap_size = cro_rss.shape[0], ob_rss.shape[0], ob_rss.shape[1]
    diff_rss = np.zeros((cro_size, ob_size, ap_size))

    for i in range(cro_size):
        for j in range(ob_size):
            diff_rss[i, j] = cro_rss[i] - ob_rss[j]

    return diff_rss


class FPDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        ob_data = load_csv(config['ob_data_path'], header=False)
        ob_data_mean = ob_data.groupby([0, 1]).mean().reset_index().values
        config['ob_size'] = len(ob_data_mean)
        self.ob_rss = ob_data_mean[:, 2:]

        cro_rss = load_csv(config['cro_data_path'], header=True).values
        self.diff_rss = process_cro_data(cro_rss, self.ob_rss)
        self.diff_rss_tensor = torch.FloatTensor(self.diff_rss / config['min_rss'])

        self.v_data = load_csv(config['v_data_path'], header=True).values

    def __len__(self):
        return len(self.diff_rss_tensor)

    def __getitem__(self, index):
        return self.diff_rss_tensor[index]


def train_model(config, logger, dataset):
    model = TFAutoencoder(input_dim=20, d_model=8, dim_feedforward=128, nhead=4, e_num_layers=4, d_num_layers=2).to(config['device'])

    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size1'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate1'])

    cur_patience = 0
    best_loss = sys.float_info.max

    logger.info(f"============================== Pre-training ==============================")
    for epoch in range(config['epoch1']):
        model.train()
        total_loss = 0.0

        for input_diff_rss in dataloader:
            input_diff_rss = input_diff_rss.to(config['device'])

            eps = 1 / config['min_rss']
            pert = eps * torch.FloatTensor(input_diff_rss.size()).normal_(mean=0, std=config['sigma']).to(config['device'])

            input = input_diff_rss + pert
            target = input_diff_rss - pert
            output = model(input, mode=TFAutoencoderMode.TRAIN1)

            loss = F.mse_loss(output * config['min_rss'], target * config['min_rss'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            cur_patience = 0
        else:
            cur_patience += 1
            if cur_patience == config['patience']:
                break

    logger.info(f"============================== Fine-tuning ==============================")
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size2'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate2'])

    for epoch in range(config['epoch2']):
        model.train()
        total_loss = 0.0

        for input_diff_rss in dataloader:
            input_diff_rss = input_diff_rss.to(config['device'])

            eps = 1 / config['min_rss']
            pert1 = eps * torch.FloatTensor(input_diff_rss.size()).normal_(mean=0, std=1).to(config['device'])
            pert2 = eps * torch.FloatTensor(input_diff_rss.size()).normal_(mean=0, std=1).to(config['device'])

            input1 = input_diff_rss + pert1
            input2 = input_diff_rss + pert2

            com_input = torch.cat((input1, input2), dim=0)
            com_output = model(com_input, mode=TFAutoencoderMode.TRAIN2)
            output1, output2 = torch.split(com_output, input_diff_rss.size(0), dim=0)

            loss = F.mse_loss(output1 * config['min_rss'], output2 * config['min_rss'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    return model.state_dict()


def exp(config):
    config["ob_data_path"] = f"./dataset/{config['name']}/sample/ob_p{config['percent']}_l{config['layout']}.csv"
    compression_saved_dir_path = create_dir(f"./dataset/{config['name']}/compression_saved/p{config['percent']}_l{config['layout']}")
    logger = create_logger(f"{compression_saved_dir_path}/out.log")

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(config)

    dataset = FPDataset(config)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dict = train_model(config, logger, dataset)

    torch.save(state_dict, f"{compression_saved_dir_path}/model.pth")


def run_ssb():
    config = DatasetConfig.ssb

    config['percent'], config['layout'] = 30, 0

    config['epoch1'] = 100
    config['epoch2'] = 10
    config['batch_size1'] = 50
    config['batch_size2'] = 50
    config['learning_rate1'] = 0.01
    config['learning_rate2'] = 0.001
    config['patience'] = 10

    config['sigma'] = 7

    exp(config)


if __name__ == '__main__':
    run_ssb()
