import sys
import json
import argparse
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from evaluation_utils import (
    load_json,
    execute_sql,
    package_sqls,
    sort_results,
    print_data,
)


def result_callback(result):
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect
):
    res = int(predicted_sql == ground_truth)
    result = {
        "sql_idx": idx,
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth,
        "res": res
    }
    return result


def run_sqls_parallel(
    sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite"
):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
                sql_dialect,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def compute_acc_by_diff(exec_results, diff_json_path, detail_results_path=None, detail_scores_path=None):
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []
    cateogoried_results = {}

    for i, content in enumerate(contents):
        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])

        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])

        if content["difficulty"] == "challenging":
            try:
                challenging_results.append(exec_results[i])
            except:
                print(i)
        
        if content['db_id'] not in cateogoried_results:
            cateogoried_results[content['db_id']] = []
        cateogoried_results[content['db_id']].append(exec_results[i])

    simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results)
    moderate_acc = sum([res["res"] for res in moderate_results]) / len(moderate_results)
    challenging_acc = sum([res["res"] for res in challenging_results]) / len(
        challenging_results
    )
    all_acc = sum(results) / num_queries
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]

    if detail_results_path is not None:
        with open(detail_results_path, "w") as f:
            json.dump(cateogoried_results, f, indent=4)
    if detail_scores_path is not None:
        detail_scores = {}
        for k, v in cateogoried_results.items():
            detail_scores[k] = sum([res["res"] for res in v]) / len(v)
            detail_scores[k] = detail_scores[k] * 100
        with open(detail_scores_path, "w") as f:
            json.dump(detail_scores, f, indent=4)

    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--output_log_path", type=str, default="SQLite")
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode='pred'
    )
    # generate ground truth sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path,
        args.db_root_path,
        mode="gt",
    )

    query_pairs = list(zip(pred_queries, gt_queries))

    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        sql_dialect=args.sql_dialect,
    )
    exec_result = sort_results(exec_result)
    print("start calculate EM")
    detail_results_path = args.output_log_path.split(".")[0] + "_EM_detail_results.json"
    detail_scores_path = args.output_log_path.split(".")[0] + "_EM_detail_scores.json"
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result, args.diff_json_path, detail_results_path=detail_results_path, detail_scores_path=detail_scores_path
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc] 
    print_data(score_lists, count_lists, metric="EM",result_log_file=args.output_log_path)
    print(
        "==========================================================================================="
    )
    print(f"Finished EM evaluation for {args.sql_dialect} on Mini Dev set")
    print("\n\n")
