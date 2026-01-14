# 基于强化学习的 AI 玩转超级马里奥

本项目基于强化学习方法，实现一个智能体在《超级马里奥兄弟（Super Mario Bros）》游戏环境中自主学习游戏策略。智能体通过与游戏环境交互，不断优化策略，以完成关卡目标。

## 一、项目简介

《超级马里奥兄弟》是经典的平台跳跃类游戏，具有连续状态空间、高维视觉输入和延迟奖励等特点，是强化学习算法验证的典型应用场景。本项目基于 `gym-super-mario-bros` 游戏环境，采用 PPO（Proximal Policy Optimization）算法，结合卷积神经网络（CNN）对游戏画面进行特征提取，实现 AI 自动玩马里奥的完整流程。

在硬件资源受限的条件下，项目重点关注系统流程的完整性与可运行性，不追求最优通关效果。

## 二、核心技术

- 强化学习：PPO（Proximal Policy Optimization）
- 深度学习框架：PyTorch
- 游戏环境：gym-super-mario-bros
- 状态表示：CNN 处理游戏画面
- 开发语言：Python

## 三、项目结构说明

MARIO/
├── MarioGame       # 游戏本体
├── src/            # 核心代码（环境、模型、算法、训练）
├── config.py       # 配置文件
├── scripts/        # 运行脚本（训练 / 评估）
├── checkpoints/    # 模型保存
├── tests/          # 单元测试
├── train.py        # 训练模型
└── play.py         # 运行效果展示

## 四、运行说明

1. 安装依赖
pip install -r requirements.txt

2. 训练模型
bash scripts/train.sh

3. 模型评估
bash scripts/evaluate.sh

## 五、实验说明

由于训练过程耗时较长，实验主要验证系统流程的正确性和强化学习训练的基本效果，包括奖励变化趋势及智能体行为变化情况。