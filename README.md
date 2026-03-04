# SelfAttnScoring-MPDocVQA
Official Implementation for ICDAR2024 paper ["Multi-Page Document Visual Question Answering using Self-Attention Scoring Mechanism"](https://arxiv.org/pdf/2404.19024)


## Dataset

Please find the MP-DocVQA dataset in [RRC Task 4](https://rrc.cvc.uab.es/?ch=17&com=tasks). More details can be found in [Ruben's GitHub repo](https://github.com/rubenpt91/MP-DocVQA-Framework).

Once you've acquired the dataset and placed it in your folder, be sure to update lines 9-10 in the `dataset.py` file accordingly.


## Train the model

All the hyperparameters can be modified within the `train.py`. To train the model, just do `python train.py`.


## Weights

The well trained weights for the scoring module can be found in `scoring_pix2struct.model.ANLS0.6199`.


## Benchmark

Please find the leaderboard [HERE](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4), and you can find this method named "(OCR-Free) Retrieval-based Baseline".


## Citation

If you find our work helpful for your research or use it as a baseline model, please cite our paper as follows:

```bibtex
@inproceedings{kang2024multi,
  title={Multi-Page Document Visual Question Answering using Self-Attention Scoring Mechanism},
  author={Kang, Lei and Tito, Rub{\`e}n and Valveny, Ernest and Karatzas, Dimosthenis},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2024},
  organization={Springer}
}
```



SelfAttnScoring-MPDocVQA/
│
├── 📁 configs/                    # 配置文件目录
│   └── config.py                  # 训练/测试的所有超参数配置
│
├── 📁 data/                       # 数据相关目录
│   ├── dataset.py                 # 数据集加载和预处理
│   ├── create_extreme_valid_npy.py # 数据集创建脚本
│   ├── imdbs/                     # 数据索引文件
│   │   ├── imdb_train.npy        # 训练集索引
│   │   ├── imdb_val.npy          # 验证集索引
│   │   └── imdb_test.npy         # 测试集索引
│   └── images/                    # 原始图像数据
│       └── (MP-DocVQA数据集图像)
│
├── 📁 models/                     # 模型定义目录
│   └── prob_model.py             # 概率评分模块定义
│
├── 📁 scripts/                    # 训练和测试脚本目录
│   ├── train/                    # 训练脚本
│   │   ├── train_ddp.py         # 分布式训练（多GPU）
│   │   └── run_train.sh         # 训练启动脚本
│   └── test/                     # 测试脚本
│       ├── test.py              # 简单测试脚本
│       ├── test_model.py        # 模型评估脚本
│       └── test_multi_gpu.py    # 多GPU测试脚本
│
├── 📁 utils/                      # 工具函数目录
│   ├── metrics.py                # 评估指标（ANLS等）
│   ├── seed.py                   # 随机种子设置
│   └── util_log.py               # 日志记录工具
│
├── 📁 checkpoints/                # 模型检查点目录
│   ├── best_model/               # 最佳模型
│   │   └── scoring_pix2struct.model.ANLS0.6199
│   └── weights/                  # 训练过程中的权重
│       ├── pix2struct-1.model
│       ├── pix2struct-2.model
│       └── pix2struct-3.model
│
├── 📁 pretrained/                 # 预训练模型目录
│   └── pix2struct-docvqa-base/   # Google预训练模型
│       └── (从HuggingFace下载的模型文件)
│
├── 📁 logs/                       # 训练日志目录
│   └── (训练过程的日志文件)
│
├── 📁 assets/                     # 静态资源目录
│   └── fonts/                    # 字体文件
│       └── arial.ttf
│
├── 📁 docs/                       # 文档目录
│   └── (项目文档、笔记等)
│
├── 📄 README.md                   # 项目说明文档
├── 📄 LICENSE                     # 许可证
├── 📄 requirements.txt            # Python依赖
└── 📄 .gitignore                  # Git忽略文件
