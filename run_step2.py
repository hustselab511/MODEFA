import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import visdom
from torch import optim
from torch.utils.data import DataLoader

from config import DatasetConfig
from model.unet import UNet
from util import load_csv, create_dir, create_logger, reshape_fp_map, inverse_reshape_fp_map, save_csv


def add_noise(config, fp_map_rss, sigma, exclude_pos):
    noise = np.random.normal(0, sigma, fp_map_rss.shape)
    noised_fp_map_rss = fp_map_rss + noise

    for pos in exclude_pos:
        x, y = pos
        noised_fp_map_rss[:, y, x] = fp_map_rss[:, y, x]

    noised_fp_map_rss = np.clip(noised_fp_map_rss, config['min_rss'], 0)

    return noised_fp_map_rss


def process_labeled_data(data):
    pos = data[:, :2]
    rss = data[:, 2:]

    unique_pos, indices = np.unique(pos, axis=0, return_inverse=True)

    rss_means = np.array([rss[indices == i].mean(axis=0) for i in range(len(unique_pos))])

    return unique_pos, rss_means


class FPDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        floor_plan_data = load_csv(config['floor_plan_path'], header=False).values
        floor_plan = floor_plan_data[:, 2]
        self.floor_plan = floor_plan.reshape((config['m'], config['n'])).T
        self.floor_plan_tensor = torch.FloatTensor(self.floor_plan)

        fp_map_data = load_csv(config['step1_fp_map_path'], header=False).values
        self.fp_map_rss = fp_map_data[:, 2:]
        self.fp_map_pos = fp_map_data[:, :2]
        fp_map_rss_2d = reshape_fp_map(config, self.fp_map_rss)
        self.fp_map_rss_tensor = torch.FloatTensor(fp_map_rss_2d[np.newaxis, :, :, :] / config['min_rss'])

        labeled_data = load_csv(config['labeled_data_path'], header=False).values
        self.labeled_pos, self.labeled_rss = process_labeled_data(labeled_data)
        self.labeled_rss_tensor = torch.FloatTensor(self.labeled_rss / config['min_rss'])

        augmented_fp_map_rss_list = []
        for i in range(config['noise_datasize']):
            augmented_fp_map_rss = add_noise(config, fp_map_rss_2d, 1, self.labeled_pos)
            augmented_fp_map_rss_list.append(augmented_fp_map_rss)
        self.augmented_fp_map_rss_tensor_list = torch.FloatTensor(np.array(augmented_fp_map_rss_list) / config['min_rss'])

        self.v_data = load_csv(config['v_data_path'], header=True).values

    def __len__(self):
        return len(self.augmented_fp_map_rss_tensor_list)

    def __getitem__(self, index):
        return self.augmented_fp_map_rss_tensor_list[index]


def evaluate(config, dataset, model):
    fp_map_rss_tensor = dataset.fp_map_rss_tensor.to(config['device'])
    model.eval()
    with torch.no_grad():
        output_fp_map_rss = model(fp_map_rss_tensor).cpu().numpy() * config['min_rss']
    new_fp_map_rss = inverse_reshape_fp_map(config, output_fp_map_rss[0])
    new_fp_map = np.concatenate((dataset.fp_map_pos, new_fp_map_rss), axis=1)

    return new_fp_map


def cal_loss2(output_fp, plan):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(output_fp.device)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(output_fp.device)

    _output_fp = output_fp.view(-1, 1, output_fp.shape[2], output_fp.shape[3])

    grad_x = F.conv2d(_output_fp, sobel_x, padding=1)
    grad_y = F.conv2d(_output_fp, sobel_y, padding=1)
    grad_total = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    plan = plan.unsqueeze(0).unsqueeze(1).expand_as(_output_fp)

    inner_grad_total = grad_total[:, :, 1:-1, 1:-1]
    inner_plan = plan[:, :, 1:-1, 1:-1]

    min_grad = inner_grad_total.min()
    max_grad = inner_grad_total.max()
    normalized_grad = (inner_grad_total - min_grad) / (max_grad - min_grad + 1e-6)

    border_loss = torch.mean(inner_plan * normalized_grad)

    loss = 1 - border_loss

    return loss


def cal_loss1(config, out, tar, labeled_pos, labeled_rss):
    denoising_mask = torch.ones_like(out)
    for pos in labeled_pos:
        x, y = pos
        denoising_mask[:, :, y, x] = 0

    denoising_out = out * denoising_mask
    denoising_tar = tar * denoising_mask

    denoising_loss = F.mse_loss(denoising_out * config['min_rss'], denoising_tar * config['min_rss'])

    fidelity_mask = torch.zeros_like(out)
    for pos in labeled_pos:
        x, y = pos
        fidelity_mask[:, :, y, x] = 1

    fidelity_out = out * fidelity_mask
    fidelity_tar = torch.zeros_like(tar)
    for i, pos in enumerate(labeled_pos):
        x, y = pos
        fidelity_tar[:, :, y, x] = labeled_rss[i]

    fidelity_loss = F.mse_loss(fidelity_out * config['min_rss'], fidelity_tar * config['min_rss'])

    return denoising_loss, fidelity_loss



def train_model(config, logger, vis, dataset):
    model = UNet(config['ap_size'], False).to(config['device'])

    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size1'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate1'])

    res_dict = {
        "fp_map": None,
    }

    cur_patience = 0
    best_loss = sys.float_info.max

    logger.info(f"============================== Pre-training ==============================")
    for epoch in range(config['epoch1']):
        model.train()
        total_loss = 0.0

        for input_fp_map_rss in dataloader:
            input_fp_map_rss = input_fp_map_rss.to(config['device'])
            labeled_rss = dataset.labeled_rss_tensor.to(config['device'])

            eps = 1 / config['min_rss']
            pert = eps * torch.FloatTensor(input_fp_map_rss.size()).normal_(mean=0, std=config['sigma']).to(config['device'])

            input = input_fp_map_rss + pert
            target = input_fp_map_rss - pert
            output = model(input)

            denoising_loss, fidelity_loss = cal_loss1(config ,output, target, dataset.labeled_pos, labeled_rss)
            loss = denoising_loss + fidelity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader): .3f}")

        if total_loss < best_loss:
            best_loss = total_loss
            cur_patience = 0
        else:
            cur_patience += 1
            if cur_patience == config['patience']:
                break


    logger.info(f"============================== Fine-tuning ==============================")
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size2'], shuffle=True)
    for param in model.inc.parameters():
        param.requires_grad = False
    for param in model.down1.parameters():
        param.requires_grad = False
    for param in model.down2.parameters():
        param.requires_grad = False
    for param in model.down3.parameters():
        param.requires_grad = False

    decoder_params = list(model.up3.parameters()) + \
                     list(model.up2.parameters()) + \
                     list(model.up1.parameters()) + \
                     list(model.outc.parameters())
    optimizer = optim.Adam(decoder_params, lr=config['learning_rate2'])

    for epoch in range(config['epoch2']):
        model.train()
        total_loss = 0.0

        for input_fp_map_rss in dataloader:
            input_fp_map_rss = input_fp_map_rss.to(config['device'])
            floor_plan = dataset.floor_plan_tensor.to(config['device'])

            output_fp_map = model(input_fp_map_rss)

            loss = cal_loss2(
                output_fp_map * config['min_rss'],
                floor_plan
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        new_fp_map = evaluate(config, dataset, model)
        res_dict['fp_map'] = new_fp_map

        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.3f}")

    return res_dict


def exp(config):
    vis = visdom.Visdom(env="step2")

    config["labeled_data_path"] = f"./dataset/{config['name']}/sample/p{config['percent']}_l{config['layout']}.csv"
    step1_fp_map_dir_path = f"./dataset/{config['name']}/fp_map/step1/p{config['percent']}_l{config['layout']}"
    config['step1_fp_map_path'] = f"{step1_fp_map_dir_path}/step1.csv"

    step2_fp_map_dir_path = create_dir(f"./dataset/{config['name']}/fp_map/step2/p{config['percent']}_l{config['layout']}")
    logger = create_logger(f"{step2_fp_map_dir_path}/out.log")

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(config)

    dataset = FPDataset(config)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    res_dict = train_model(config, logger, vis, dataset)

    save_csv(res_dict['fp_map'], f"{step2_fp_map_dir_path}/step2.csv")


def run_ssb():
    config = DatasetConfig.ssb

    config['draw_ap_idx'] = 7
    config['percent'], config['layout'] = 30, 0

    config['noise_datasize'] = 500
    config['sigma'] = 7

    config['epoch1'] = 100
    config['epoch2'] = 10
    config['batch_size1'] = 50
    config['batch_size2'] = 50
    config['learning_rate1'] = 0.005
    config['learning_rate2'] = 0.0001
    config['patience'] = 10

    exp(config)


if __name__ == '__main__':
    run_ssb()
