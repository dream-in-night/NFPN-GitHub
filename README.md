# NFPN-GitHub
# 基于mmdetection，先安装mmdetection配置好环境。
# 训练方法：

# 1 NFPN

1.将本项目中mmdet/model/neck中的nfpn.py以及__init__.py文件复制到mmdetection/mmdet/neck下，替换__init__.py文件

python tools/train.py config/faster_rcnn/nfpn.py
# 2 NFPN + DROI 

2.将本项目中mmdet/model/detectors中的twostage.py文件替换到mmdetection/mmdet/detectors下的twostage.py

3.将本项目中mmdet/model/roi_heads中的standard_roi_head.py文件复替换mmdetection/mmdet/roi_heads中的standard_roi_head

4.将本项目中mmdet/model/roi_heads/roi_extractors文件夹替换复替换mmdetection/mmdet/roi_heads中的/roi_extractors文件夹

# 3，NFPN + DROI + RFPN


**表格 1 在Lisa数据集上的实验**

| Method                           | mAP      | AP50     | AP75     | APS      | APM      | APL      | backbone   |
|----------------------------------|----------|----------|----------|----------|----------|----------|------------|
|                                  |          |          |          |          |          |          |            |
| Faster RCNN                      | 65.2     | 76.1     | 74.8     | 66.8     | 71.0     | 70.5     | Res50-FPN  |
| BFP                              | 63.9     | 75.9     | 74.0     | 63.4     | 71.2     | 77.9     | ResN50-FPN |
| BiFPN                            | 58.4     | 70.9     | 69.8     | 55.4     | 65.1     | 83.5     | Res50-FPN  |
| BiFPN\*2                         | 49.2     | 59.9     | 58.4     | 49.1     | 56.6     | 78.5     | Res50-FPN  |
| NFPN(ours)                       | 66.1     | 77.5     | 75.8     | 65.9     | 72.1     | 75.7     | Res50-FPN  |
| **Double ROI+ NFPN(ours)**       | **67.4** | **78.8** | **77.5** | **68.0** | **74.3** | **76.0** | Res50-FPN  |
| NFPN\*2+ Double ROI(ours)        | 66.0     | 78.1     | 76.4     | 66.2     | 72.5     | 80.7     | Res50-FPN  |
| **NFPN +递归+ Double ROI(ours)** | **67.8** | **79.0** | **77.7** | **68.4** | **74.0** | **81.1** | Res50-FPN  |


**表格 2**  **在 VOC07+12上的实验**

| Method         | mAP      | backbone    |
|----------------|----------|-------------|
|                |          |             |
| Faster RCNN    | 73.2     | VGG-16      |
| Faster RCNN    | 79.5     | Res-50      |
| **NFPN(ours)** | **81.3** | **Res-50**  |
| Faster RCNN    | 76.4     | Res-101     |
| **NFPN(ours)** | **82.4** | **Res-101** |

