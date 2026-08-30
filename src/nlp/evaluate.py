#!/usr/bin/env python3
"""
Evaluate NLP extraction results against manually verified ground truth.
Calculates precision, recall, and F1 for entities and relationships.
"""

import json
import os
from collections import defaultdict


def normalize_text(t):
    return t.strip().lower()


def compute_entity_metrics(extracted, ground_truth):
    """Compute entity-level precision, recall, F1."""
    # Group ground truth by (type, normalized text, doc_id)
    gt_set = set()
    gt_by_type = defaultdict(set)
    for doc in ground_truth["documents"]:
        for ent in doc["entities"]:
            key = (normalize_text(ent["text"]), ent["type"], doc["doc_id"])
            gt_set.add(key)
            gt_by_type[ent["type"]].add(key)
    
    # Group extracted by same key
    ex_set = set()
    ex_by_type = defaultdict(set)
    for ent in extracted:
        key = (normalize_text(ent["text"]), ent["type"], ent["source_document"])
        ex_set.add(key)
        ex_by_type[ent["type"]].add(key)
    
    tp = gt_set & ex_set
    fp = ex_set - gt_set
    fn = gt_set - ex_set
    
    precision = len(tp) / len(ex_set) if ex_set else 0
    recall = len(tp) / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per-type
    per_type = {}
    for etype in sorted(set(list(gt_by_type.keys()) + list(ex_by_type.keys()))):
        gt_t = gt_by_type.get(etype, set())
        ex_t = ex_by_type.get(etype, set())
        tp_t = gt_t & ex_t
        p = len(tp_t) / len(ex_t) if ex_t else 0
        r = len(tp_t) / len(gt_t) if gt_t else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_type[etype] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": len(tp_t), "fp": len(ex_t - gt_t), "fn": len(gt_t - tp_t),
            "extracted": len(ex_t), "ground_truth": len(gt_t)
        }
    
    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "total_extracted": len(ex_set),
            "total_ground_truth": len(gt_set)
        },
        "per_type": per_type
    }


def compute_relationship_metrics(extracted, ground_truth):
    """Compute relationship-level precision, recall."""
    gt_set = set()
    for doc in ground_truth["documents"]:
        for rel in doc["relationships"]:
            key = (normalize_text(rel["source"]), normalize_text(rel["target"]), rel["type"], doc["doc_id"])
            gt_set.add(key)
    
    ex_set = set()
    for rel in extracted:
        key = (normalize_text(rel["source"]), normalize_text(rel["target"]), rel["type"], rel["source_document"])
        ex_set.add(key)
    
    tp = gt_set & ex_set
    fp = ex_set - gt_set
    fn = gt_set - ex_set
    
    precision = len(tp) / len(ex_set) if ex_set else 0
    recall = len(tp) / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per type
    gt_by_type = defaultdict(set)
    ex_by_type = defaultdict(set)
    for key in gt_set:
        gt_by_type[key[2]].add(key)
    for key in ex_set:
        ex_by_type[key[2]].add(key)
    
    per_type = {}
    for rtype in sorted(set(list(gt_by_type.keys()) + list(ex_by_type.keys()))):
        gt_t = gt_by_type.get(rtype, set())
        ex_t = ex_by_type.get(rtype, set())
        tp_t = gt_t & ex_t
        p = len(tp_t) / len(ex_t) if ex_t else 0
        r = len(tp_t) / len(gt_t) if gt_t else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_type[rtype] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "true_positives": len(tp_t), "extracted": len(ex_t), "ground_truth": len(gt_t)
        }
    
    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "total_extracted": len(ex_set),
            "total_ground_truth": len(gt_set)
        },
        "per_type": per_type
    }


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    with open(os.path.join(base, "data", "nlp_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(base, "data", "nlp_ground_truth.json")) as f:
        ground_truth = json.load(f)
    
    entity_metrics = compute_entity_metrics(results["all_entities"], ground_truth)
    relationship_metrics = compute_relationship_metrics(results["all_relationships"], ground_truth)
    
    evaluation = {
        "disclaimer": "Evaluation performed on synthetic demonstration data only. Does not represent real-world FIR processing performance.",
        "data_source": "Synthetic FIR documents with manually verified ground-truth annotations",
        "entity_extraction": entity_metrics,
        "relationship_extraction": relationship_metrics
    }
    
    results["evaluation"] = evaluation
    results["_meta"]["evaluation_completed"] = True
    
    with open(os.path.join(base, "data", "nlp_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=== Evaluation Results ===")
    print(f"\nEntity Extraction:")
    print(f"  Precision: {entity_metrics['overall']['precision']}")
    print(f"  Recall:    {entity_metrics['overall']['recall']}")
    print(f"  F1:        {entity_metrics['overall']['f1']}")
    print(f"  TP={entity_metrics['overall']['true_positives']} FP={entity_metrics['overall']['false_positives']} FN={entity_metrics['overall']['false_negatives']}")
    print(f"\n  Per-type breakdown:")
    for etype, m in entity_metrics["per_type"].items():
        print(f"    {etype}: P={m['precision']} R={m['recall']} F1={m['f1']} ({m['tp']}/{m['ground_truth']})")
    
    print(f"\nRelationship Extraction:")
    print(f"  Precision: {relationship_metrics['overall']['precision']}")
    print(f"  Recall:    {relationship_metrics['overall']['recall']}")
    print(f"  F1:        {relationship_metrics['overall']['f1']}")
    print(f"  TP={relationship_metrics['overall']['true_positives']} FP={relationship_metrics['overall']['false_positives']} FN={relationship_metrics['overall']['false_negatives']}")
    print(f"\n  Per-type breakdown:")
    for rtype, m in relationship_metrics["per_type"].items():
        print(f"    {rtype}: P={m['precision']} R={m['recall']} F1={m['f1']} ({m['true_positives']}/{m['ground_truth']})")
    
    return evaluation


if __name__ == "__main__":
    main()
