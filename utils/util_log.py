import os
import sys
import torch

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs import config


class LOG:
    def __init__(self, file_name, epoch):
        log_dir = os.path.join(project_root, config.LOG_DIR)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  
        self.file = open(os.path.join(log_dir, f'{file_name}_res_epoch-{epoch}.log'), 'a')
        self.file.write('Question ID | Document ID | Find Page | GT | Prediction\n')

    def write(self, batch, preds, slices_gt, slices):
        gts = batch['answers']
        qids = batch['question_id']
        imids = batch['image_names']
        pageids = batch['answer_page_idx']
        for c, i in enumerate(slices_gt):
            self.file.write(f"{qids[i]}\t| {imids[i].split('/')[-1].split('.')[0]}\t| {'True' if slices[c] == i else '----'}\t| {gts[i]} | {preds[i]} \n")


def save_model(model, epoch):
    weight_dir = os.path.join(project_root, config.WEIGHT_DIR)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    torch.save(model.state_dict(), os.path.join(weight_dir, f'pix2struct-{epoch}.model'))
