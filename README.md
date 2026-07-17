# DoS Attack Scheduling Simulation

论文 *DoS attack schedules for remote state estimation under multi-sensor round-robin protocol* 数值实验的独立、简化 Python 复现程序。

- Journal: *Systems & Control Letters* 204 (2025) 106190
- DOI: `10.1016/j.sysconle.2025.106190`
- Python: 3.12

本仓库只保存自查和复现所需的核心程序、参数及说明。Monte Carlo 原始样本、CSV、图片、日志、虚拟环境和缓存均在本地运行时生成，不纳入版本控制。

## 快速开始

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_full.bat
```

完整模式执行 100,000 次 Monte Carlo 仿真。快速检查：

```powershell
.\run_quick.bat
```

也可直接运行：

```powershell
.\.venv\Scripts\python.exe .\run_all.py --trials 100000
.\.venv\Scripts\python.exe .\validate_outputs.py
```

## 核心文件

- `config.json`：论文参数、补充假设和固定随机种子。
- `run_all.py`：全部仿真与制图入口。
- `src/model.py`：系统、信道、Kalman 协方差和状态轨迹模型。
- `src/experiment.py`：Monte Carlo 实验、CSV/NPZ导出和运行清单。
- `src/plotting.py`：图2-6绘图程序。
- `validate_outputs.py`：结果完整性和趋势验证。
- `requirements.txt`：最小Python依赖。
- `run_full.bat`、`run_quick.bat`：Windows一键运行脚本。

## 本地生成的数据

运行后自动产生以下目录，但不会上传到GitHub：

- `data/`：10万次逐次代价样本、图2-6 CSV源数据、状态/噪声数组、参数环境记录及SHA-256清单。
- `figures/`：图2-6的PNG和PDF。
- `logs/`：运行摘要。

这些输出可用于自查表中的四类材料：

1. 数值仿真源代码及运行说明（Python）。
2. 系统与信道参数、随机种子及运行环境记录。
3. 10万次 Monte Carlo 仿真逐次代价样本及图2-4源数据。
4. 状态轨迹、估计结果、噪声序列及图5-6源数据。

## 独立复现假设

论文未公开作者原始代码、随机种子、初始协方差以及功率到信道成功概率的完整换算公式。为保证程序可运行、可重复，本项目采用并在 `config.json` 中记录以下假设：

- 初始协方差为单位阵。
- 攻击状态下信道对角成功率采用 `alpha = ps / (ps + 5 * pa)` 的代理公式。
- 图2-3采用前置加权平均代价，第 `k` 步权重为 `T-k+1`。
- 智能传感器成功上传本地滤波结果时采用0.7的协方差缩减系数。
- 图2-4使用论文给出的 `A11=2`。
- 仅图5-6的有界示意轨迹使用 `A11=0.92`，并在前5个攻击时刻强制丢包。
- 所有随机过程均采用固定种子。

因此，本仓库应称为“独立仿真复现程序”，不能声称是论文作者投稿时的原始程序或原始实验记录。
