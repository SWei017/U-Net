# F1 score function
import torch


def f1(pred, target, dilated_target):
    eps = 1e-8
    pred = pred.data.to('cpu')
    target = target.data.to('cpu')
    dilated_target = dilated_target.data.to('cpu')

    # Accuracy
    acc = (pred == dilated_target) * 1
    acc = torch.sum(acc, dim=2) / acc.size(2)
    acc = torch.sum(acc, dim=2) / acc.size(2)
    acc = torch.sum(acc) / acc.size(0)

    pred = (pred > 0) * 1
    tp1 = pred * dilated_target
    tp2 = pred * target
    tp = torch.sum(tp1, dim=2)
    tp = torch.sum(tp, dim=2)
    tp_act = torch.sum(tp2, dim=2)
    tp_act = torch.sum(tp_act, dim=2)

    tp_fp = torch.sum(pred, dim=2)
    tp_fp = torch.sum(tp_fp, dim=2)
    tp_fn = torch.sum(target, dim=2)
    tp_fn = torch.sum(tp_fn, dim=2)
    fn = tp_fn - tp_act

    prec = tp / (tp_fp + eps)
    recall = tp / (tp + fn + eps)
    # recall = tp / (tp_fn + eps)
    f1_score = 2.0 * (prec * recall) / (prec + recall + eps)

    prec = torch.sum(prec) / prec.size(0)
    recall = torch.sum(recall) / recall.size(0)
    f1_score = torch.sum(f1_score) / f1_score.size(0)

    return acc.item(), prec.item(), recall.item(), f1_score.item()
