# Post-Training Quantization和Quantization-Aware Training
| |PTQ|	QAT|
|---|---|---|
|全称|Post-Training Quantization（训练后量化）|Quantization-Aware Training（量化感知训练）|
|时机|模型训练完之后做|训练过程中做|
|需要的资源|推理 + 一小批校准数据|完整重训流程 + 训练数据|
|成本|低（几小时）|高（按训练时长算）|
|典型精度|8bit 无损 ~ 4bit 掉点|4bit 也能保住精度|
|工业界用途|绝大多数 8bit 部署|低位宽 / 精度敏感场景|
## PTQ 完整流程
训练好的 fp32 模型
   1.  拿校准集跑推理（只推理不训练）
   2.  统计每层激活的 min/max（或 KL 散度，见 SUMMARY 第四节）
   3.  定每层 scale / zero_point
   4.  权重直接按 min/max 量化（权重是常量，不用校准）
   5.  导出 int8 模型

## QAT 为什么精度更高
QAT 在训练图里插入 fake quantize 节点：

    训练时:  y_q = round(y / scale) * scale      # 前向走量化数值
    反向时:  梯度直接穿过 round（STE: straight-through estimator）  # round 不可导, 假装它是恒等
关键点：模型在训练过程中就"见过"量化噪声，梯度会带着"量化会怎么伤我"的信息去调整权重——权重学会了在量化后依然鲁棒。而 PTQ 是"事后诸葛亮"：模型完全没见过量化噪声，遇到 outlier 分布直接崩。

## 8bit分水岭
8bit 量化误差小（6dB/bit，48dB 起），PTQ 通常够用；
降到 4bit 必须 QAT——4bit 只有 24dB 余量，PTQ 的 outlier 敏感直接暴露。

## 动态量化（dynamic）
是PTQ内部的子类
## 静态量化（static）
也是PTQ内部的子类

|   |动态量化	|静态量化|
|---|---|---|
|权重|	离线量化好|	离线量化好|
|激活scale|	运行时实时统计（每帧算 min/max）|	离线校准好，烘焙进模型|
|速度|	慢（统计本身耗时）	|快（无运行时开销）|
|使用|	大模型 LLM 部署常用	|NPU/DSP 部署主流（QNN 就是）|
实验里的 scale_x 如果运行时算就是动态，离线算好就是静态.

## 总结一句话
+ PTQ = 训练后拿数据校准 scale（便宜、8bit 够用）；
+ QAT = 训练中模拟量化（贵、4bit 必须）；
+ 动态/静态是 PTQ 内部"激活 scale 何时定"的区分。