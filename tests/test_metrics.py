import math

from toolrank.eval.metrics import aggregate, category_macro, evaluate_query


def test_perfect_single_relevant():
    m = evaluate_query(["a", "b", "c"], {"a": 1}, ks=(1, 3))
    assert m["NDCG@1"] == 1.0 and m["NDCG@3"] == 1.0
    assert m["Recall@1"] == 1.0 and m["Precision@1"] == 1.0
    assert m["Precision@3"] == 1 / 3
    assert m["MAP@3"] == 1.0
    assert m["Comprehensiveness@1"] == 1.0


def test_trec_eval_linear_gain_and_log2_discount():
    # relevant doc at rank 2 only: DCG = 1/log2(3); ideal = 1/log2(2) = 1
    m = evaluate_query(["x", "a", "y"], {"a": 1}, ks=(3,))
    assert math.isclose(m["NDCG@3"], 1 / math.log2(3))
    assert math.isclose(m["MAP@3"], 1 / 2)


def test_graded_relevance_uses_linear_gain():
    # ranked: rel2 first, rel1 second -> ideal order, NDCG = 1
    m = evaluate_query(["a", "b"], {"a": 2, "b": 1}, ks=(2,))
    assert math.isclose(m["NDCG@2"], 1.0)
    # swapped: DCG = 1 + 2/log2(3); IDCG = 2 + 1/log2(3)
    m2 = evaluate_query(["b", "a"], {"a": 2, "b": 1}, ks=(2,))
    assert math.isclose(m2["NDCG@2"], (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_recall_and_map_divide_by_total_relevant_and_comprehensiveness():
    m = evaluate_query(["a", "x", "y"], {"a": 1, "b": 1, "c": 1}, ks=(3,))
    assert math.isclose(m["Recall@3"], 1 / 3)
    assert math.isclose(m["MAP@3"], (1 / 1) / 3)
    assert m["Comprehensiveness@3"] == 0.0
    m2 = evaluate_query(["b", "a", "c"], {"a": 1, "b": 1, "c": 1}, ks=(3,))
    assert m2["Recall@3"] == 1.0 and m2["Comprehensiveness@3"] == 1.0


def test_no_relevant_docs_is_zero_not_nan():
    m = evaluate_query(["a"], {}, ks=(1,))
    assert m["NDCG@1"] == 0.0 and m["Recall@1"] == 0.0 and m["Comprehensiveness@1"] == 0.0


def test_aggregate_is_mean_over_queries():
    rows = [{"NDCG@10": 1.0, "Recall@10": 0.0}, {"NDCG@10": 0.0, "Recall@10": 1.0}]
    assert aggregate(rows) == {"NDCG@10": 0.5, "Recall@10": 0.5}
    assert aggregate([]) == {}


def test_category_macro_is_mean_of_category_means_of_task_means():
    # the ToolRet paper's Average: web = mean(1.0, 0.0) = 0.5, code = 0.2, overall = 0.35;
    # a plain mean over the three tasks would give 0.4
    per_task = {"a": {"NDCG@10": 1.0}, "b": {"NDCG@10": 0.0}, "c": {"NDCG@10": 0.2}}
    per_cat, overall = category_macro(per_task, {"a": "web", "b": "web", "c": "code"})
    assert per_cat == {"code": {"NDCG@10": 0.2}, "web": {"NDCG@10": 0.5}}
    assert math.isclose(overall["NDCG@10"], 0.35)
