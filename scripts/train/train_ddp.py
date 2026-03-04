import os
import sys
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from tqdm import tqdm
import random
import numpy as np

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from data.dataset import loadData
from utils.metrics import Evaluator
from models.prob_model import ProbModule
from utils import util_log
from utils.seed import set_seed
from configs import config

# ============================================================
# 从配置文件导入参数
# ============================================================
FIX_SEED = config.FIX_SEED
SEED = config.SEED
EARLY_STOP = config.EARLY_STOP
BATCH_SIZE = config.BATCH_SIZE
NUM_THREAD = config.NUM_THREAD
MAX_EPOCHS = config.MAX_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
lr_milestone = config.lr_milestone
lr_gamma = config.lr_gamma

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# ============================================================
# 辅助函数
# ============================================================
def collate_batch(batch):
    """批处理数据整理函数"""
    new_batch = {k: [dic[k] for dic in batch for _ in (0, 1)] 
                 for k in batch[0] if k not in ['image_names', 'image_patches', 'patches_masks', 'rela_probs']}
    
    final_image_names = []
    final_patches = []
    final_masks = []
    final_rela_probs = []
    
    for item in batch:
        final_image_names.extend(item['image_names'])
        final_patches.extend(item['image_patches'])
        final_masks.extend(item['patches_masks'])
        final_rela_probs.extend(item['rela_probs'])
    
    new_batch['image_names'] = final_image_names
    new_batch['image_patches'] = torch.tensor(np.array(final_patches, dtype=np.float32))
    new_batch['patches_masks'] = torch.tensor(np.array(final_masks, dtype=np.float32))
    new_batch['rela_probs'] = torch.tensor(np.array(final_rela_probs, dtype=np.float32))
    
    return new_batch

def rand_choice_answer(batch_answers):
    return [random.choice(answer) for answer in batch_answers]

def obtain_slice(probs):
    slices = []
    count = len(probs)
    for i in range(0, count, 2):
        idx = i if probs[i] > probs[i+1] else i+1
        slices.append(idx)
    return slices

# ============================================================
# 单卡训练主函数
# ============================================================
def train(start_epoch=0):
    if FIX_SEED:
        set_seed(SEED)
    
    print(f"\n{'='*60}")
    print("Single GPU Training")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("Loading datasets...")
    data_train, data_val = loadData()
    
    dataloader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=False
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        data_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=False
    )
    
    print(f"✓ Train: {len(data_train)} samples, {len(dataloader_train)} batches")
    print(f"✓ Val: {len(data_val)} samples, {len(dataloader_val)} batches")
    
    # 加载模型
    print("\nLoading models...")
    model_path = os.path.join(project_root, config.MODEL_NAME)
    if not os.path.exists(model_path):
        model_path = config.MODEL_NAME  # 使用在线模型名称
    
    processor = Pix2StructProcessor.from_pretrained(
        model_path,
        use_fast=False
    )
    
    model = Pix2StructForConditionalGeneration.from_pretrained(
        model_path
    )
    
    # 单卡: Encoder 和 ProbModule 都在同一张卡
    encoder = model.encoder.to(DEVICE)
    probModule = ProbModule().to(DEVICE)
    
    print(f"✓ Encoder on {DEVICE}: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")
    print(f"✓ ProbModule on {DEVICE}: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")
    
    # 加载检查点
    if start_epoch > 0:
        checkpoint_path = os.path.join(project_root, config.WEIGHT_DIR, f'pix2struct-{start_epoch}.model')
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        probModule.load_state_dict(checkpoint)
        print(f'✓ Checkpoint loaded: epoch {start_epoch}')
    
    encoder.eval()
    probModule.train()
    
    optimizer = torch.optim.Adam(
        params=probModule.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestone,
        gamma=lr_gamma
    )
    
    evaluator = Evaluator(case_sensitive=False)
    
    best_cor = 0
    best_epoch = 0
    max_num = 0
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        import time
        start_time = time.time()
        print(f"\n--- Epoch {epoch}/{MAX_EPOCHS} ---")
        
        loss, cor = train_one_epoch(
            processor, encoder, probModule, optimizer, evaluator, dataloader_train, epoch
        )
        
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        print(f'\n#### TRAIN-{epoch} -- Loss: {loss:.4f}, Acc: {cor*100:.2f}%, lr: {lr:.6f}, time: {elapsed:.2f}s')
        print(f'    {DEVICE}: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB')
        
        scheduler.step()
        
        # 验证
        print(f"\nValidating...")
        loss_val, cor_val = evaluate(
            processor, encoder, probModule, evaluator, dataloader_val, epoch
        )
        print(f'VALID-{epoch} -- Loss: {loss_val:.4f}, Acc: {cor_val*100:.2f}%')
        
        if cor_val > best_cor:
            best_cor = cor_val
            best_epoch = epoch
            max_num = 0
            util_log.save_model(probModule, epoch)
            print(f'✓ Best model saved: {cor_val*100:.2f}%')
        else:
            max_num += 1
            print(f'No improvement ({max_num}/{EARLY_STOP})')
        
        if max_num >= EARLY_STOP:
            print(f'\n[EARLY STOP] Best: {best_cor*100:.2f}% @ epoch {best_epoch}')
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best: {best_cor*100:.2f}% @ epoch {best_epoch}")
    print(f"{'='*60}\n")

# ============================================================
# 单卡训练 epoch 函数
# ============================================================
def train_one_epoch(processor, encoder, probModule, optimizer, evaluator, dataloader, epoch):
    total_loss = 0
    cor_page_counts = 0
    count = 0
    
    probModule.train()
    encoder.eval()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
    
    for batch in pbar:
        question = batch['question']
        answers = rand_choice_answer(batch['answers'])
        answers_tensor = processor.tokenizer(answers, return_tensors='pt', padding=True).to(DEVICE)
        
        rela_probs = batch['rela_probs'].to(DEVICE)
        image_patches = batch['image_patches'].to(DEVICE)
        patches_masks = batch['patches_masks'].to(DEVICE)
        
        with torch.no_grad():
            outputs = encoder(image_patches, patches_masks, output_attentions=True)
            features = outputs.last_hidden_state
        
        enc_feat = torch.permute(features, (0, 2, 1))
        probs = probModule(enc_feat, patches_masks)
        rela_loss = evaluator.mse_loss(probs, rela_probs)
        
        optimizer.zero_grad()
        rela_loss.backward()
        optimizer.step()
        
        slices = obtain_slice(probs.detach().cpu())
        slices_gt = obtain_slice(rela_probs.detach().cpu())
        correct_page_count = len(set(slices) & set(slices_gt))
        
        total_loss += rela_loss.item()
        cor_page_counts += correct_page_count
        count += len(slices)
        
        pbar.set_postfix({
            'loss': f'{total_loss/count:.4f}',
            'acc': f'{cor_page_counts/count:.4f}'
        })
    
    return total_loss/count, cor_page_counts/count

# ============================================================
# 单卡验证函数
# ============================================================
def evaluate(processor, encoder, probModule, evaluator, dataloader, epoch):
    total_loss = 0
    cor_page_counts = 0
    count = 0
    
    encoder.eval()
    probModule.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", ncols=100):
            rela_probs = batch['rela_probs'].to(DEVICE)
            image_patches = batch['image_patches'].to(DEVICE)
            patches_masks = batch['patches_masks'].to(DEVICE)
            
            outputs = encoder(image_patches, patches_masks, output_attentions=True)
            features = outputs.last_hidden_state
            
            enc_feat = torch.permute(features, (0, 2, 1))
            probs = probModule(enc_feat, patches_masks)
            rela_loss = evaluator.mse_loss(probs, rela_probs)
            
            slices = obtain_slice(probs.cpu())
            slices_gt = obtain_slice(rela_probs.cpu())
            correct_page_count = len(set(slices) & set(slices_gt))
            
            total_loss += rela_loss.item()
            cor_page_counts += correct_page_count
            count += len(slices)
    
    return total_loss/count, cor_page_counts/count

# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    train()
