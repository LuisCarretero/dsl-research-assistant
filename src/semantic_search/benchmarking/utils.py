from typing import List, Dict

def score_predictions(gt_refs: List[str], pred_refs: List[str]) -> Dict[str, float]:
    assert len(gt_refs) > 0, "Ground truth references must not be empty"
    
    gt_set = set(map(lambda x: x.lower(), gt_refs))
    pred_set = set(map(lambda x: x.lower(), pred_refs))
    
    true_positives = len(gt_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'prec': precision, 'rec': recall, 'f1': f1}

def calc_metric_at_levels(true: List[str], pred: List[str], levels: List[int], pref_level: int) -> Dict[str, float]:
    metrics = {'ref_cnt': len(pred)}
    for level in levels:
        metrics.update({f'{k}_lvl{level}': v for k, v in score_predictions(true, pred[:level]).items()})
    metrics.update({f'{k}_preflvl{pref_level}': v for k, v in score_predictions(true, pred[:pref_level]).items()})
    return metrics
