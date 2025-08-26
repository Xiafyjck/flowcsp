这是一个比赛项目，利用仿真的PXRD生成准确的晶体结构，比赛的[详细说明](docs/competition)需要完整阅读

## 设计说明

1. 利用最新的 mean flow 生成框架，[code](docs/repo/meanflow)，[literature](docs/literature/meanflow)
2. 采用等变性模型 equiformer，[code](docs/repo/equiformer)，[literature](docs/literature/equiformer)
3. 仿真PXRD计算是一个[开源的过程](src/PXRDSimulator.py)，但是无法逆向。因此我们可以在逐步采样的时候实时计算PXRD作为中途条件

- 实时计算的PXRD应该是不参数反向传播的，可以看作模型会向代理计算PXRD，然后直接获得PXRD
- 模型输入：初始条件，niggli晶胞原子数量 comp， 单一样本维度是52维，52代表最多有52个原子，每个维度用atom.Z表示，后续的0就表示padding或者叫mask， 晶体仿真11051维pxrd以及采样过程中实时仿真计算出来的pxrd 11051维度，注意这里是有两个pxrd谱；因为是生成模型，模型的输出同时也会作为下一轮采样的输入
- 模型输出：[batch_size, 3 + 52, 3]，前3个维度是晶格参数，采用 3*3 向量矩阵表示（为了后续和equiformer兼容，放弃长度和角度表示），后52维度是与niggli晶胞原子数量维度一致的分数坐标 frac_coords，注意frac_coords每个维度是跟原子数量comp一一对应的，同时满足一个置换不变性，假如有一个置换群元素g，model(g(comp),...) = g(model(comp, ...))
- 模型参数用 z = model(z, t, r, conditions)表示， z代表模型输出，t和r是两个时间步，剩余的则为条件conditions

## 环境说明

本机同时也是训练环境，双卡A100的Linux容器

比赛测试服务器也就是推理环境，单卡T4的Linux容器



## 技术栈说明

- 采用 uv 管理 python 运行环境，运行python脚本请执行 source .venv/bin/activate && python，安装模块请执行 uv add，安装本地模块请执行 uv pip install -e .
- 采用pytorch-lightning管理训练
- 所有模块都采用较新的版本，遇到兼容问题不应该回退版本，而是和用户讨论解决办法
- conf预计采用hydra管理，但现在是开发阶段，采用硬编码更为合适
- 训练时请采用双卡策略加速

### 最终技术栈

- uv：python 运行环境管理
- hydra + python-dotenv: 配置管理
- pytorch: 训练框架
- wandb：日志记录


## 调参说明

- 我创建了一个[超小数据集](data/small_dataset_1000.pkl)（只有1000个数据点且数据结构和标准数据集一致，所以不用查看里面的具体数据结构）来验证模型是否有过拟合能力，如果模型不能过拟合，那么就是失败的模型



## 目前遇到的问题及解决办法

- [x] 可能是linux容器的原因，matplot一定不要用中文

- [ ] 逐步采样的时候实时计算PXRD，逐步采样是从一个先验分布到后验分布到过程，中间的插值可能无法计算出PXRD，所以无法计算出的应该要想办法丢弃改条件
- [ ] equiformer可能难以构建，因此前期我们先用参数量在150M左右的transformer打通流程

- [ ] meanflow是新的生成框架，有很多技术要点需要看论文，例如怎么使用中间使用的jvp技术和梯度停止技术



## 先跑通整个pipeline，后优化

- 舍弃配置管理，不引入任何 hydra 和 python-dotenv 代码，默认硬编码 + 命令行参数即可
- 不采用任何日志记录模块，利用 print, tqdm, p-tqdm, rich 实时反馈在 pytorch-lightning的训练过程中即可

## 数据流程

1. scripts/preprocess.py 首先对原始数据文件进行预处理得到 .pkl 文件
2. dataset类直接处理 .pkl 文件
3. 训练
4. 利用部分[公开的比赛原数据](data/A_sample)测试模型效果，
4. 一个[ipynb文件](submission.ipynb)，给定[比赛规定的数据格式](docs/data/test_v3/A)，处理代码请参考 docs/repo/cdvae/scripts/competition_evaluate.py，模型推理，后处理（能量优化，重新生成），生成 submission.csv 到指定目录

## 交互偏好
1. 测试通过后，列出可以删除的中间代码文件，主动询问用户是否可以删除
2. 请对 docs 使用 tree 命令，获得足够的上下文
3. 代码请加入详细的中文注释和具体数据流讲解
4. docs是只读文件夹，不能修改其中内容

## 推荐的文件组织形式

  src/
  ├── networks/         # 所有网络定义（独立模块，与其他模块无关）
  │   ├── __init__.py     # 统一接口 + 网络注册机制
  │   ├── readme.md       # 接口文档
  │   ├── transformer.py  # Transformer实现 + 数据适配
  │   └── equiformer.py   # Equiformer实现 + 数据适配（主要是会涉及到pytorch geometric data）
  ├── flows/           # 所有flows定义（独立模块，与其他模块）
  │   ├── __init__.py   
  │   ├── readme.md       # 接口文档
  │   ├── meanflow.py    
  │   └── cfm.py          #利用pytorchcfm实现的标准cfm
  ├── trainer.py       # Lightning模块 + 训练逻辑（网络无关，flows无关）
  ├── data.py          # 数据加载（返回原始字典，网络，flows无关）
  ├── metrics.py       # 评估指标
  └── pxrd_simulator.py  # PXRD仿真

  train.py             # 训练启动脚本（只负责配置和启动）
  inference.ipynb      # 功能并不是比赛提交脚本，而是利用[公开的比赛原数据](data/A_sample)测试模型效果，用matplot分析

  scripts/              
  test/                 # AI会话中需要的创建的测试文件全部放这里，会话结束用户会手动删除

  ⚠️ 重要规则

  1. 不要轻易修改接口
  2. 数据转换在网络内：PyG等特殊格式转换放在网络的 prepare_batch 中
  3. 保持文件独立：修改一个功能尽量只改一个文件

  🐛 问题定位指南

  - 数据加载问题 → 检查 data.py
  - 网络前向传播问题 → 检查 networks/具体网络.py
  - 训练不收敛 → 检查 flow 的损失计算
  - 训练流程问题 → 检查 trainer.py
  - 采样质量问题 → 检查 flow 的 sample 方法

  🚀 快速开始

  - 使用Transformer训练
  python train.py --network transformer --flow cfm

  - 使用Equiformer训练
  python train.py --network equiformer --flow meanflow

  - 切换网络只需要改 --network 参数！切换flow只需要改 --flow 参数！