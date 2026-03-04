"""
配置文件 - 集中管理所有训练参数
"""

# ==================== 分布式训练设置 ====================
WORLD_SIZE = 1  # GPU数量        [[[[已改]]]]
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

# ==================== 训练超参数 ====================
# 每个GPU的batch size，总batch size = BATCH_SIZE * WORLD_SIZE
BATCH_SIZE = 2  # 4卡总共64    [[[[已改]]]]

NUM_THREAD = 2  # 每个进程的DataLoader worker数
MAX_EPOCHS = 4
LEARNING_RATE = 1e-4
lr_milestone = list(range(1, 21))
lr_gamma = 0.8

# ==================== 其他设置 ====================
FACIL = False
FIX_SEED = True
SEED = 42
EARLY_STOP = 4

# ==================== 路径设置 ====================
# 相对于项目根目录的路径
IMDB = "data/imdbs/"
IMG = "data/images/"

# ==================== 模型设置 ====================
MODEL_NAME = 'pretrained/pix2struct-docvqa-base'  # 或线上下载路径 'google/pix2struct-docvqa-base'
FONT_PATH = 'assets/fonts/arial.ttf'

# ==================== 日志设置 ====================
LOG_DIR = 'logs'
WEIGHT_DIR = 'checkpoints/weights'  # 统一到checkpoints目录

# ==================== 优化器设置 ====================
OPTIMIZER_BETAS = (0.9, 0.98)
OPTIMIZER_EPS = 1e-9

# ==================== 内存优化设置 ====================
# 如果遇到OOM，可以尝试：
# 1. 减少BATCH_SIZE（例如改为8）
# 2. 启用gradient_checkpointing
# 3. 使用mixed precision training
USE_MIXED_PRECISION = False  # 是否使用混合精度训练
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数