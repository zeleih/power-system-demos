# 基于PMU的gSCR非迭代解析辨识

本案例复现并扩展以下论文提出的广义短路比解析辨识方法：

> Z. Han, P. Ju, H. Li, and Y. Liu, “Analytical Identification Method of
> Generalized Short-Circuit Ratio Using Phasor Measurement Units,” *IET
> Generation, Transmission & Distribution*, 2025.
> <https://doi.org/10.1049/gtd2.70026>

仓库只提供论文链接和引用信息，不上传论文PDF。

## 内容

- 三端口教学算例：用于快速理解解析辨识全过程；
- CEPRI/EPRI 36节点论文算例的两个版本：
  - 归档PSASP兼容相量验证，并提供真实PSASP端口相量CSV导入接口；
  - 转换为ANDES的算例及归档时域相量验证；
- ANDES IL200同步机—IBR混合系统扩展；
- 论文中的多时刻累计、非迭代解析辨识代码；
- 可复核的PMU数据、矩阵、表格、图和JSON结果。

本项目复现论文的方法链路和相近结果，不宣称逐采样点复现论文未公开的
原始PSASP暂态波形。

## 解析方法

保留端口的相邻PMU增量满足：

\[
\Delta I_k=\bar{Y}\Delta U_k.
\]

令复对称端口导纳为：

\[
\bar{Y}=G+jB,\qquad \bar{Y}=\bar{Y}^{\mathsf T}.
\]

对多时刻最小二乘目标函数关于全部独立的 \(G_{ij}\)、\(B_{ij}\)
求偏导并令其为零，得到未知量数目相同的解析方程。程序逐批累计：

\[
C_U=\sum_k\Delta U_k^{\mathrm H}\Delta U_k,
\qquad
C_{UI}=\sum_k\Delta U_k^{\mathrm H}\Delta I_k,
\]

随后一次求解实数 \(G/B\) 方程，不需要初值、步长和迭代停止判据。
详见 [docs/method.md](docs/method.md)。

## 三层算例

### 三端口教学算例

给定已知复对称导纳，生成相量并重新辨识。当前导纳相对误差约为
`2.3e-16`，gSCR误差约为`8.9e-16`。

### CEPRI36论文算例

Bus1–Bus8统一作为八个聚合IBR端口，容量来自PSASP算例的机组额定MVA：

```text
1880, 706, 882, 235, 637.5, 100, 286, 388.399994 MVA
```

- 论文理论值：`0.1701`；
- PSASP公开参考网络：`0.17140950`；
- ANDES Bus30故障直接值：`0.17140950`；
- ANDES Bus30故障辨识值：`0.17140985`。

PSASP版支持两种入口：使用仓库内归档相量验证解析结果，或者将PSASP实际
导出的Bus1–Bus8电压、电流相量CSV送入同一解析辨识器。许可状态未确认的
PSASP原始记录不随仓库发布；获准使用者可在本机补充记录以运行完整重建。

### IL200扩展算例

IL200包含38台同步机和11个IBR。49个在线电源端口首先用于辨识外部无源
网络；之后以同步机次暂态电抗Norton导纳终接38个同步机端口，最终只在
11个IBR端口上形成容量矩阵并计算gSCR。

- 标准短路口径，不计负荷：`0.81099796`；
- ANDES时域恒阻抗负荷匹配理论值：`0.89356187`；
- PMU解析辨识值：`0.89356446`。

两种口径对应不同网络模型，不能混为同一个基准。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe scripts\run_toy3.py
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe scripts\execute_notebook.py
.\.venv\Scripts\python.exe scripts\run_cepri36_psasp.py
.\.venv\Scripts\python.exe scripts\run_cepri36_andes.py
.\.venv\Scripts\python.exe scripts\run_il200.py
.\.venv\Scripts\python.exe scripts\validate_all.py
.\.venv\Scripts\python.exe scripts\build_manifest.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

更多说明：

- [CEPRI36双版本](cases/cepri36/README.md)
- [可执行方法教程](notebooks/analytical_gscr_walkthrough.ipynb)
- [IL200混合系统](cases/il200/README.md)
- [端口定义](docs/port_definition.md)
- [验证结果](docs/results.md)
- [PSASP相量CSV接口](docs/psasp_export.md)
- [第三方来源与许可](THIRD_PARTY.md)
- [参考文件哈希清单](results/reference/artifact_manifest.json)
