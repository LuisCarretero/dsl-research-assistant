from typing import Tuple, List

def score_predictions(gt_refs: List[str], pred_refs: List[str]) -> Tuple[float, float, float]:
    assert len(gt_refs) > 0, "Ground truth references must not be empty"
    
    gt_set = set(map(lambda x: x.lower(), gt_refs))
    pred_set = set(map(lambda x: x.lower(), pred_refs))
    
    true_positives = len(gt_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
