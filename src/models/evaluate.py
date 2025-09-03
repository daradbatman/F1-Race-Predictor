import pandas as pd
from src.data.open_F1_service import fetch_latest_session_results

def evaluate_model():
    # Step 1: Fetch latest race results
    latest_result = fetch_latest_session_results()

    # Fill missing positions (DNFs, DNS, DSQ) with 31
    df_results = latest_result[["driver_number", "position", "dnf", "dns", "dsq"]]
    df_results["position"] = df_results["position"].fillna(31)
    df_results["position"] = df_results["position"].apply(lambda x: int(x) if x != 0 else 31)
    df_results = df_results.rename(columns={"position": "actual_finish"})

    # Step 2: Load predictions log
    predictions_log = pd.read_csv("data/predictions/predictions_log.csv")

    # Step 3: Merge predictions with actual results
    df_eval = predictions_log.merge(
        df_results,
        on="driver_number",
        how="inner"
    )

    # Step 4: Compute evaluation metrics
    spearman = df_eval[["predicted_rank", "actual_finish"]].corr(method="spearman").iloc[0,1]
    kendall = df_eval[["predicted_rank", "actual_finish"]].corr(method="kendall").iloc[0,1]

    # Per-driver accuracy (how many finishing positions exactly correct)
    per_driver_accuracy = (df_eval["predicted_rank"].astype(int) == df_eval["actual_finish"].astype(int)).mean()

    # Exact order accuracy (whole race lineup matches prediction)
    exact_order_accuracy = int(
        df_eval.sort_values("predicted_rank")["driver_number"].tolist() ==
        df_eval.sort_values("actual_finish")["driver_number"].tolist()
    )

    eval_report = {
        "spearman_rank_corr": spearman,
        "kendall_rank_corr": kendall,
        "per_driver_accuracy": per_driver_accuracy,
        "exact_order_accuracy": exact_order_accuracy
    }

    # Step 5: Save evaluation results (append to log for history)
    pd.DataFrame([eval_report]).to_csv(
        "data/evaluations/eval_log.csv",
        mode="a",
        header=not pd.io.common.file_exists("data/evaluations/eval_log.csv"),
        index=False
    )

    return df_eval, eval_report