import pandas as pd
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr, kendalltau
from src.data.open_F1_service import fetch_latest_session_results
import logging

logging.basicConfig(
    filename="logs/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def evaluate_model():
    # Step 1: Fetch latest race results
    latest_result = fetch_latest_session_results()

    # Fill missing positions (DNFs, DNS, DSQ) with 21
    df_results = latest_result[["meeting_key","driver_number", "position", "dnf", "dns", "dsq"]]
    df_results["position"] = df_results["position"].fillna(21)

    # Step 2: Load predictions log
    predictions_log = pd.read_csv("data/predictions/prediction_log.csv")
    df_results.loc[df_results["dnf"] | df_results["dns"] | df_results["dsq"], "position"] = 21

    merged = predictions_log.merge(df_results, on="driver_number")

    true_scores = 21 - merged["position"].to_numpy()
    pred_scores = 21 - merged["predicted_rank"].to_numpy()

    ndcg = ndcg_score([true_scores], [pred_scores], k=10)
    tau, _ = kendalltau(true_scores, pred_scores)
    rho, _ = spearmanr(true_scores, pred_scores)

    actual_winner = merged.loc[merged["position"].idxmin(), "driver_name"]
    predicted_winner = merged.loc[merged["predicted_rank"].idxmin(), "driver_name"]
    winner_correct = actual_winner == predicted_winner
    
    logging.info(f"Evaluation for race:")
    logging.info(f"  NDCG@10: {ndcg:.4f}")
    logging.info(f"  Kendall Tau: {tau:.4f}")
    logging.info(f"  Spearman: {rho:.4f}")
    logging.info(f"  Winner predicted correctly? {winner_correct} "
          f"(Pred={predicted_winner}, Actual={actual_winner})")

    return {
        "ndcg@10": ndcg,
        "kendall_tau": tau,
        "spearman": rho,
        "winner_correct": winner_correct,
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner
    }