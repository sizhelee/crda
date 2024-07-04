import json
import math

with open("scanrefer.jsonl", "r", encoding='utf-8') as f:
    ori_info = json.load(f)
info = {}
for item in ori_info:
    info[item["item_id"]] = item

with open("./test/scanrefer-test-mask-ori/preds/val_outs.json", "r") as f:
    obj_pred = json.load(f)
with open("./test/scanrefer-test/preds/val_outs.json", "r") as f:
    ori_pred = json.load(f)

def aggregation(x, y):
    return (x ** 2.45) * (y ** 1)
    # return x
    # return y

cnt = 0
correct = 0
iou25, iou50 = 0, 0
top3_correct = 0
for key in ori_pred.keys():
    ori_pred_item = ori_pred[key]
    obj_pred_item = obj_pred[key]
    gt_item = info[key]

    ori_pred_logits = ori_pred_item["obj_logits"]
    obj_pred_logits = obj_pred_item["obj_logits"]
    
    idx = 0
    max_logits = 0
    for i in range(len(ori_pred_logits)):
        logits = aggregation(math.e ** ori_pred_logits[i], math.e ** obj_pred_logits[i])
        # logits = math.e ** ori_pred_logits[i]
        if logits > max_logits:
            max_logits = logits
            idx = i
    # max3 = []
    # for i in range(2):
    #     max_tmp = max(ori_pred_logits)
    #     idx_tmp = ori_pred_logits.index(max_tmp)
    #     # max3.append(int(ori_pred_item["obj_ids"][idx_tmp]))
    #     max3.append(idx_tmp)
    #     ori_pred_logits[idx_tmp] = -1e9
    # max_logits = 0
    # idx = max3[0]
    # for i in max3:
    #     logits = math.e ** obj_pred_logits[i]
    #     if logits > max_logits:
    #         max_logits = logits
    #         idx = i

    # if gt_item["target_id"] == int(ori_pred_item["obj_ids"][idx]):
    #     correct += 1
    if ori_pred_item["obj_ious"][idx] >= 0.25:
        iou25 += 1
    if ori_pred_item["obj_ious"][idx] >= 0.5:
        iou50 += 1
    # if gt_item["target_id"] in max3:
    #     top3_correct += 1
    # cnt += 1


print("iou@25:", iou25/cnt, "iou@50:", iou50/cnt)
# print(correct / cnt)