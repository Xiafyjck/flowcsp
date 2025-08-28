这是一个比赛项目，利用仿真的PXRD生成准确的晶体结构，比赛的[详细说明](docs/competition)需要完整阅读

## 设计说明

1. 利用最新的 mean flow 生成框架，[code](docs/repo/meanflow)，[literature](docs/literature/meanflow)
2. 采用等变性模型 equiformer，[code](docs/repo/equiformer)，[literature](docs/literature/equiformer)
3. 仿真PXRD计算是一个[开源的过程](src/PXRDSimulator.py)，但是无法逆向。因此我们可以在逐步采样的时候实时计算PXRD作为中途条件


- 模型输入：初始条件，niggli晶胞原子数量 comp， 单一样本维度是60维，60代表最多有60个原子，每个维度用atom.Z表示，后续的0就表示padding或者叫mask， 晶体仿真11501维pxrd；因为是生成模型，模型的输出同时也会作为下一轮采样的输入

- 模型输出：[batch_size, 3 + 60, 3]，前3个维度是晶格参数，采用 3*3 向量矩阵表示（为了后续和equiformer兼容，放弃长度和角度表示），后60维度是与niggli晶胞原子数量维度一致的分数坐标 frac_coords，注意frac_coords每个维度是跟原子数量comp一一对应的，同时满足一个置换等变性，假如有一个置换群元素g，model(g(comp),...) = g(model(comp, ...))

- 模型参数用 z = model(z, t, r, conditions)表示， z代表模型输出，t和r是两个时间步，剩余的则为条件conditions

- 原子编号和坐标的置换等变性通过数据增强实现：晶体中原子编号与坐标必须一一对应，打乱顺序时需保持这种对应关系（置换等变性）。训练时随机生成置换矩阵P（用F.one_hot(randperm(n))），padding到60维后同时作用于comp和frac_coords，确保原子-坐标同步置换。这比改造成等变网络简单多了。


- [ ] 晶格参数SO3(晶格参数是三个向量，起点都是坐标系零点，所以已经约化了平移变换)变换对分数坐标的不变性，这一步目前靠数据增强实现，达到的效果是 model_frac(g_so3(lattice), ...)= model_frac(lattice, ...)，（这主要是因为晶格参数是由3个向量表示，如果采用6自由度表示就没这个问题）； 晶格参数SO3变换对条件comp,pxrd的不变性；晶格参数去噪网络的SO3等变性，这一步由 equiformer 实现，达到的效果是 model_lattice(g_so3(lattice), ...)= g_so3(model_lattice(lattice, ...))，这一步只需要对初始数据的晶格参数施加SO3变换，然后Flow Matching线性插值来增强该等变性。因此这三种等变性的数据增强方式都是同一种方式，对初始数据的晶格参数施加一个SO3变换即可

- 晶格参数需要实现归一化，分数坐标不需要，因为分数坐标本身就是在(0,1)的范围，这样z更好训练, 这个步骤应该是由 flow.sample实现对用户透明，由 src/normalizer.py提供对应工具和静态数据



- loss 设计需要利用mask处理变长分数坐标，padding部分不能参与计算
## 环境说明

本机同时也是训练环境，双卡A100的Linux容器

比赛测试服务器也就是推理环境，单卡T4的Linux容器

编写代码请务必考虑训练策略是DDP, 之前因为没把所有参数放在同一device从而花费了很多时间debug




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

- [ ] equiformer可能难以构建，因此前期我们先用参数量在150M左右的transformer打通流程

- [ ] meanflow是新的生成框架，有很多技术要点需要看论文，例如怎么使用中间使用的jvp技术和梯度停止技术

- [ ] 遇到训练NaN问题，可以用用PyTorch的异常检测定位： torch.autograd.set_detect_anomaly(True) 或环境变量：TORCH_ANOMALY_ENABLED=1

- [ ] 目前我搜集的训练数据集太大了，所以为了训练方便，我写了warmup脚本，把数据转换成npz格式

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
5. 使用rich库优化终端输出：进度条实时覆盖更新，每个epoch的训练指标永久保留，平衡可读性与token效率。

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
  inference.ipynb      # 实现基于RWP指标的迭代优化推理流程。初始推理后，循环评估PXRD匹配质量，对不满足RWP<0.15 的样本进行能量优化→Rietveld精修→批量重新生成，直至达到终止条件（5小时时限/单样本10次上限/全部合格），
                          每次重新生成增量更新CIF格式的submission.csv
  evaluate.ipynb       # 解析CIF格式submission.csv、与ground truth对比计算RMSD、统计分析和可视化

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