# NFPN-GitHub
# 基于mmdetection，先安装mmdetection配置好环境。
# 训练方法：
# 1 NFPN
python tools/train.py config/faster_rcnn/nfpn.py
# 2 NFPN + DROI 
# 3，NFPN + DROI + RFPN
因为需要修改two_stage.py，bbox_head.py，single_roi_extractor和base_roi_extractor等文件，需要另建一个虚拟环境，mmdet和config在mmdet2中，即将上传。
