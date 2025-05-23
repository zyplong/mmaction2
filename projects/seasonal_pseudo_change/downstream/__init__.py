# projects/seasonal_pseudo_change/downstream/__init__.py
# 一定要把这俩子模块都 import 进来，否则 custom_imports 会找不到
from .models.load_pt import PTFeatureDataset
from .pipelines.load_pt_feature import LoadPTFeature
# 如果你也用到了 seco_resnet，也顺便 import 一下
from .models.seco_resnet import SeCoResNet