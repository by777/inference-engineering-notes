# per-channel量化的原因

## 1.什么是per-channel量化？
per-channel 量化：给每个 channel 一个独立的 scale，而不是整个张量共用一个。

## 2.为什么要per-channel量化？
看 4D 激活 [N, C, H, W]，比如 [1, 64, 8, 8]：

N=1 batch, C=64 通道, H=8 高, W=8 宽
+ per-tensor：整个 1×64×8×8 共用一个 scale（被最大的那个 channel 拉大，小 channel 精度丢）。
+ per-channel：64 个 channel 各算各的 scale，scales 数组长度 = 64。

## 3.代码中的axes = (0, 2, 3)什么意思？
```python
    scales = np.max(np.abs(x), axis=(0, 2, 3)) / (q_max - q_min)
```
0， 2， 3表示在N,H,W三个维度上取max，只保留C维：
输入 x: [1, 64, 8, 8]
对 axis=(0,2,3) 取 max → 输出 [64]    # 每个 channel 一个 max 值

## 4.scales[None, :, None, None] 是什么
scales是[64]的一维数据，这个操作把它扩展成[1, 64, 1, 1]，方便广播到原始张量的形状[1, 64, 8, 8]，每个 channel 的 scale 对应到整个 channel 的所有元素。

## 5.2D权重[C_out, C_in]怎么切

W: [64, 256]    # C_out=64 输出通道, C_in=256 输入通道
```python
scales = np.max(np.abs(W), axis=1) / 127.0    # 对 axis=1(C_in) 取 max → [64]
#        每个 C_out 一行, 取这行的 max → 该输出通道的 scale
q_pc = np.clip(np.round(W / scales[:, None]), -128, 127)  # scales[:,None]: [64,1]
x_hat_pc = q_pc * scales[:, None]
```
为什么权重按 C_out 切：因为推理时"输出通道 i 的每个输出元素"都乘的是"第 i 个输入通道的权重行"，per-C_out 恰好让每个输出通道的 scale 独立——这是硬件/推理引擎实际的 per-channel 粒度

## 6. 长尾分布是指什么？
分布形状探测是指分析张量中数值的分布特征，例如接近 0 的比例（长尾/稀疏度）和峰度（kurtosis）。这些特征可以帮助判断量化后可能的精度损失，判断"这层激活像什么分布"。
+ n_near_zero：接近 0 的比例（长尾/稀疏探测器）

```python
n_near_zero = np.mean(np.abs(flat) < 0.01 * xmax) if xmax > 0 else 1.0
#             └─ 求比例           └─ 判断"绝对值 < max 的 1%"
# flat：所有元素展平成一维
# np.abs(flat) < 0.01 * xmax：布尔数组，标记"值小于最大值的 1%"的元素
# np.mean(布尔数组)：True 比例 = 多大比例的元素挤在 0 附近

# 例子（100 个元素，max=10.0）：
# 90 个元素是 0.05, 10 个元素是 10
np.abs(flat) < 0.01*10 = 0.1  → 90 个元素 < 0.1 吗?
0.05 < 0.1 → True (90 个)  → mean = 90/100 = 0.9
near0 = 90% → 90% 的元素挤在极小的范围内 → 长尾分布。
```
**为什么和量化相关：量化 scale 由 max 决定。如果 90% 的数据只有 max 的 1%，那 scale 相对它们太大 → 这 90% 全被量化成 0 或 ±1 → 小值信息全丢。所以 near0 高 = 量化危险信号。**

## 7. kurt：峰度（分布"尾巴"有多重）
```python
kurt = ((flat - flat.mean()) ** 4).mean() / (flat.var() + 1e-12) ** 2
```
为什么和量化相关：kurt 大 = 有极端大值 → scale 被它们拉大 → 其余数据压缩精度。和 near0 是互补的双重验证——两者都指向"长尾"。

## 总结
near0 = 多大比例的元素挤在 0 附近（小值多不多），kurt = 分布尾巴有多重（大值有多极端）。两者互补刻画"长尾度"——长尾分布量化必崩（scale 被大值拉大、小值全变 0），所以这两个指标高 = 量化危险信号，实验 2 靠它们自动扫出敏感层。