from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# 加载COCO格式的参考标注和预测结果
coco = COCO(r'E:\nlp\final\CNN-RNN\coco_annotations_val_simple.json')  # 标注文件的路径
coco_res = coco.loadRes(r'E:\nlp\final\CNN-RNN\.cache_fc_val_new1.json')  # 预测结果文件的路径

# 创建COCOEvalCap对象并评估
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.evaluate()

# 输出所有评估指标
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score}')
