from typing import List, Dict

def score_predictions(gt_refs: List[str], pred_refs: List[str]) -> Dict[str, float]:
    if len(gt_refs) == 0:
        return {'prec': -1, 'rec': -1, 'f1': -1, 'jaccard': -1}
    assert len(gt_refs) > 0, "Ground truth references must not be empty"
    
    gt_set = set(map(lambda x: x.lower(), gt_refs))
    pred_set = set(map(lambda x: x.lower(), pred_refs))
    
    true_positives = len(gt_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Jaccard index (intersection over union)
    union = len(gt_set.union(pred_set))
    jaccard = true_positives / union if union > 0 else 0
    
    return {'prec': precision, 'rec': recall, 'f1': f1, 'jaccard': jaccard}

def calc_metric_at_topk(true: List[str], pred: List[str], topks: List[int]) -> Dict[str, float]:
    metrics = {'ref_cnt': len(pred)}
    for topk in topks:
        metrics.update({f'{k}_top{topk}': v for k, v in score_predictions(true, pred[:topk]).items()})
    return metrics
