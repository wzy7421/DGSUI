DGSUI-Recommendation/
│
├── data/                   # 存放 Taobao, Electronics, Steam 数据集
├── models/
│   ├── __init__.py
│   ├── layers.py           # 核心模块 (Time-aware MHSA, Orthogonal Engine)
│   └── dgsui.py            # DGSUI 主模型组装
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # R@N, H@N, C@N, B@N 计算
│   └── loss.py             # 正交损失、重建损失、香农熵损失、自适应BPR损失
│
├── main.py                 # 训练与测试主循环
├── requirements.txt        # 依赖环境
└── README.md               # 论文介绍与运行说明