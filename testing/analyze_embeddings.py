"""
Comprehensive Embedding Analysis Pipeline for `test_students` dataset.

Evaluates quality, similarity distributions, hard negatives, margins, retrieval metrics,
threshold statistics (FAR, FRR, TAR, EER, F1), identity confusion matrices, dataset diagnostics,
and exports plots, JSON summaries, CSV artifacts, and a detailed Markdown report.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.schema import pool, init_db

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EmbeddingAnalysis")

# --- Configuration & Default Paths ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")


class EmbeddingAnalyzer:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_data = []
        self.embeddings_matrix = None  # (N, D) normalized numpy array
        self.metadata = []             # List of dicts per sample
        self.sim_matrix = None         # (N, N) cosine similarity matrix
        self.results = {}

    def load_dataset(self):
        """Loads all embeddings and metadata from test_students table in a single query."""
        logger.info("Loading embeddings and metadata from `test_students` table...")
        init_db()
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, kerberos_id, data_point_number, sample_number, det_score, embedding
                    FROM test_students
                    ORDER BY kerberos_id, data_point_number, sample_number;
                """)
                rows = cur.fetchall()

        if not rows:
            raise ValueError("No records found in `test_students` table.")

        logger.info(f"Loaded {len(rows)} samples from database.")

        embeddings_list = []
        self.metadata = []

        for row in rows:
            rec_id, kerberos_id, dp_num, sample_num, det_score, emb_val = row
            
            # Parse vector string format if returned as string
            if isinstance(emb_val, str):
                emb_arr = np.fromstring(emb_val.strip("[]"), sep=",", dtype=np.float32)
            else:
                emb_arr = np.array(emb_val, dtype=np.float32)

            # L2 normalize for fast cosine similarity dot product
            norm = np.linalg.norm(emb_arr)
            if norm > 0:
                emb_arr = emb_arr / norm

            embeddings_list.append(emb_arr)
            self.metadata.append({
                "id": rec_id,
                "kerberos_id": str(kerberos_id),
                "data_point_number": int(dp_num),
                "sample_number": int(sample_num),
                "det_score": float(det_score)
            })

        self.embeddings_matrix = np.vstack(embeddings_list)
        logger.info(f"Embeddings matrix created with shape: {self.embeddings_matrix.shape}")

    def compute_pairwise_similarity(self):
        """Computes N x N cosine similarity matrix using vector dot product."""
        logger.info("Computing pairwise cosine similarity matrix...")
        start = time.time()
        # Since vectors are L2-normalized, cosine similarity is matrix multiplication
        self.sim_matrix = np.dot(self.embeddings_matrix, self.embeddings_matrix.T)
        # Clip numerical inaccuracies to [-1.0, 1.0]
        self.sim_matrix = np.clip(self.sim_matrix, -1.0, 1.0)
        logger.info(f"Similarity matrix computed in {time.time() - start:.3f} seconds.")

    def analyze_similarities_and_margins(self):
        """Computes genuine/impostor matches, hard negatives, margins, and retrieval accuracy."""
        logger.info("Analyzing genuine/impostor matches, hard negatives, and margins...")
        N = len(self.metadata)

        genuine_scores = []
        impostor_scores = []
        
        sample_results = []
        hard_negatives = []
        genuine_matches_records = []
        impostor_matches_records = []

        # Vectorized lookup structures
        kerberos_ids = np.array([m["kerberos_id"] for m in self.metadata])
        dp_numbers = np.array([m["data_point_number"] for m in self.metadata])
        sample_numbers = np.array([m["sample_number"] for m in self.metadata])

        top_1_hits = 0
        top_3_hits = 0
        top_5_hits = 0
        top_10_hits = 0

        for i in range(N):
            q_meta = self.metadata[i]
            q_kid = q_meta["kerberos_id"]

            # Masks for genuine and impostor targets
            is_same_id = (kerberos_ids == q_kid)
            is_same_sample = np.zeros(N, dtype=bool)
            is_same_sample[i] = True

            genuine_mask = is_same_id & (~is_same_sample)
            impostor_mask = ~is_same_id

            sims = self.sim_matrix[i]

            # Genuine scores for query i
            gen_indices = np.where(genuine_mask)[0]
            gen_sims = sims[gen_indices]
            genuine_scores.extend(gen_sims.tolist())

            for g_idx, g_sim in zip(gen_indices, gen_sims):
                m_meta = self.metadata[g_idx]
                genuine_matches_records.append({
                    "query_id": q_meta["id"],
                    "query_kerberos": q_kid,
                    "query_dp": q_meta["data_point_number"],
                    "query_sample": q_meta["sample_number"],
                    "matched_id": m_meta["id"],
                    "matched_kerberos": m_meta["kerberos_id"],
                    "matched_dp": m_meta["data_point_number"],
                    "matched_sample": m_meta["sample_number"],
                    "similarity": float(g_sim)
                })

            # Impostor scores for query i
            imp_indices = np.where(impostor_mask)[0]
            imp_sims = sims[imp_indices]
            impostor_scores.extend(imp_sims.tolist())

            # Best genuine and best impostor match for query i
            best_gen_sim = float(np.max(gen_sims)) if len(gen_sims) > 0 else -1.0
            worst_gen_sim = float(np.min(gen_sims)) if len(gen_sims) > 0 else -1.0

            best_imp_idx = imp_indices[np.argmax(imp_sims)] if len(imp_sims) > 0 else -1
            best_imp_sim = float(sims[best_imp_idx]) if best_imp_idx >= 0 else -1.0
            best_imp_meta = self.metadata[best_imp_idx] if best_imp_idx >= 0 else {}

            margin = best_gen_sim - best_imp_sim

            # Record hard negative for query i
            if best_imp_idx >= 0:
                hard_negatives.append({
                    "query_id": q_meta["id"],
                    "query_kerberos": q_kid,
                    "query_dp": q_meta["data_point_number"],
                    "query_sample": q_meta["sample_number"],
                    "hard_negative_id": best_imp_meta["id"],
                    "hard_negative_kerberos": best_imp_meta["kerberos_id"],
                    "hard_negative_dp": best_imp_meta["data_point_number"],
                    "hard_negative_sample": best_imp_meta["sample_number"],
                    "impostor_similarity": best_imp_sim,
                    "genuine_margin": margin
                })

            sample_results.append({
                "id": q_meta["id"],
                "kerberos_id": q_kid,
                "data_point_number": q_meta["data_point_number"],
                "sample_number": q_meta["sample_number"],
                "best_genuine_sim": best_gen_sim,
                "worst_genuine_sim": worst_gen_sim,
                "mean_genuine_sim": float(np.mean(gen_sims)) if len(gen_sims) > 0 else -1.0,
                "best_impostor_sim": best_imp_sim,
                "hard_negative_kerberos": best_imp_meta.get("kerberos_id", ""),
                "margin": margin
            })

            # Top-K Retrieval Evaluation
            # Rank all items except query i
            rank_mask = ~is_same_sample
            ranked_indices = np.where(rank_mask)[0]
            sorted_ranks = ranked_indices[np.argsort(-sims[ranked_indices])]

            # Check if any genuine item appears in top K
            top1_kids = [self.metadata[idx]["kerberos_id"] for idx in sorted_ranks[:1]]
            top3_kids = [self.metadata[idx]["kerberos_id"] for idx in sorted_ranks[:3]]
            top5_kids = [self.metadata[idx]["kerberos_id"] for idx in sorted_ranks[:5]]
            top10_kids = [self.metadata[idx]["kerberos_id"] for idx in sorted_ranks[:10]]

            if q_kid in top1_kids: top_1_hits += 1
            if q_kid in top3_kids: top_3_hits += 1
            if q_kid in top5_kids: top_5_hits += 1
            if q_kid in top10_kids: top_10_hits += 1

        self.results["sample_results"] = pd.DataFrame(sample_results)
        self.results["hard_negatives"] = pd.DataFrame(hard_negatives)
        self.results["genuine_matches"] = pd.DataFrame(genuine_matches_records)
        
        self.results["genuine_scores"] = np.array(genuine_scores, dtype=np.float32)
        self.results["impostor_scores"] = np.array(impostor_scores, dtype=np.float32)

        # Retrieval accuracy metrics
        self.results["retrieval_metrics"] = {
            "top_1_accuracy": top_1_hits / N,
            "top_3_accuracy": top_3_hits / N,
            "top_5_accuracy": top_5_hits / N,
            "top_10_accuracy": top_10_hits / N,
            "total_queries": N
        }
        logger.info(f"Top-1 Accuracy: {top_1_hits / N:.4f}, Top-5: {top_5_hits / N:.4f}")

    def analyze_data_point_consistency(self):
        """Analyzes consistency within data points (3 samples) and across data points."""
        logger.info("Evaluating Data Point Consistency (within vs across)...")
        
        within_stats = []
        across_stats = []

        unique_kids = np.unique([m["kerberos_id"] for m in self.metadata])

        for kid in unique_kids:
            kid_indices = [i for i, m in enumerate(self.metadata) if m["kerberos_id"] == kid]
            kid_dps = np.unique([self.metadata[i]["data_point_number"] for i in kid_indices])

            # Within data points
            for dp in kid_dps:
                dp_indices = [i for i in kid_indices if self.metadata[i]["data_point_number"] == dp]
                if len(dp_indices) > 1:
                    sub_sims = []
                    for a in range(len(dp_indices)):
                        for b in range(a + 1, len(dp_indices)):
                            sub_sims.append(self.sim_matrix[dp_indices[a], dp_indices[b]])
                    sub_sims = np.array(sub_sims)
                    within_stats.append({
                        "kerberos_id": kid,
                        "data_point_number": int(dp),
                        "sample_count": len(dp_indices),
                        "mean_sim": float(np.mean(sub_sims)),
                        "min_sim": float(np.min(sub_sims)),
                        "max_sim": float(np.max(sub_sims)),
                        "std_sim": float(np.std(sub_sims))
                    })

            # Across data points for identity `kid`
            if len(kid_dps) > 1:
                across_sims = []
                for i in range(len(kid_indices)):
                    for j in range(i + 1, len(kid_indices)):
                        idx_a, idx_b = kid_indices[i], kid_indices[j]
                        if self.metadata[idx_a]["data_point_number"] != self.metadata[idx_b]["data_point_number"]:
                            across_sims.append(self.sim_matrix[idx_a, idx_b])
                if across_sims:
                    across_sims = np.array(across_sims)
                    across_stats.append({
                        "kerberos_id": kid,
                        "data_points_count": len(kid_dps),
                        "mean_sim": float(np.mean(across_sims)),
                        "min_sim": float(np.min(across_sims)),
                        "max_sim": float(np.max(across_sims)),
                        "std_sim": float(np.std(across_sims))
                    })

        self.results["within_dp_consistency"] = pd.DataFrame(within_stats)
        self.results["across_dp_consistency"] = pd.DataFrame(across_stats)

    def analyze_per_identity_and_confusion(self):
        """Computes per-identity aggregate stats and builds identity confusion matrix."""
        logger.info("Computing per-identity statistics and confusion matrix...")
        
        sample_df = self.results["sample_results"]
        unique_kids = np.unique([m["kerberos_id"] for m in self.metadata])
        
        per_id_stats = []

        for kid in unique_kids:
            kid_samples = sample_df[sample_df["kerberos_id"] == kid]
            gen_records = self.results["genuine_matches"][self.results["genuine_matches"]["query_kerberos"] == kid]
            
            gen_sims = gen_records["similarity"].values if len(gen_records) > 0 else np.array([])
            hard_negs = self.results["hard_negatives"][self.results["hard_negatives"]["query_kerberos"] == kid]
            imp_sims = hard_negs["impostor_similarity"].values if len(hard_negs) > 0 else np.array([])
            margins = kid_samples["margin"].values

            per_id_stats.append({
                "kerberos_id": kid,
                "num_samples": len(kid_samples),
                "num_data_points": int(kid_samples["data_point_number"].nunique()),
                "avg_genuine_sim": float(np.mean(gen_sims)) if len(gen_sims) > 0 else 0.0,
                "min_genuine_sim": float(np.min(gen_sims)) if len(gen_sims) > 0 else 0.0,
                "max_genuine_sim": float(np.max(gen_sims)) if len(gen_sims) > 0 else 0.0,
                "std_genuine_sim": float(np.std(gen_sims)) if len(gen_sims) > 0 else 0.0,
                "avg_impostor_sim": float(np.mean(imp_sims)) if len(imp_sims) > 0 else 0.0,
                "highest_impostor_sim": float(np.max(imp_sims)) if len(imp_sims) > 0 else 0.0,
                "avg_genuine_margin": float(np.mean(margins)) if len(margins) > 0 else 0.0,
                "min_genuine_margin": float(np.min(margins)) if len(margins) > 0 else 0.0
            })

        id_df = pd.DataFrame(per_id_stats)
        # Rank identities from easiest to hardest based on average genuine margin
        id_df["difficulty_rank"] = id_df["avg_genuine_margin"].rank(ascending=False, method="min").astype(int)
        id_df = id_df.sort_values(by="difficulty_rank")
        self.results["per_identity_stats"] = id_df

        # Identity-level confusion matrix based on hard negatives
        hard_neg_df = self.results["hard_negatives"]
        confusion_pairs = hard_neg_df.groupby(["query_kerberos", "hard_negative_kerberos"]).size().reset_index(name="confusion_count")
        confusion_pairs = confusion_pairs.sort_values(by="confusion_count", ascending=False)
        self.results["confusion_pairs"] = confusion_pairs

    def evaluate_thresholds(self):
        """Sweeps thresholds across [0.0, 1.0] and computes TAR, FAR, FRR, EER, and optimal thresholds."""
        logger.info("Sweeping similarity thresholds and computing ROC/EER metrics...")
        
        gen_scores = self.results["genuine_scores"]
        imp_scores = self.results["impostor_scores"]

        thresholds = np.linspace(0.0, 1.0, 1001)
        metrics_list = []

        eer = 1.0
        eer_threshold = 0.5
        min_far_frr_diff = 1.0

        best_f1 = -1.0
        best_f1_threshold = 0.5

        for th in thresholds:
            # Genuine: >= th is True Accept (TA), < th is False Reject (FR)
            ta = np.sum(gen_scores >= th)
            fr = np.sum(gen_scores < th)
            tar = ta / len(gen_scores) if len(gen_scores) > 0 else 0.0
            frr = fr / len(gen_scores) if len(gen_scores) > 0 else 0.0

            # Impostor: >= th is False Accept (FA), < th is True Reject (TR)
            fa = np.sum(imp_scores >= th)
            tr = np.sum(imp_scores < th)
            far = fa / len(imp_scores) if len(imp_scores) > 0 else 0.0

            precision = ta / (ta + fa) if (ta + fa) > 0 else 0.0
            recall = tar
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            diff = abs(far - frr)
            if diff < min_far_frr_diff:
                min_far_frr_diff = diff
                eer = (far + frr) / 2.0
                eer_threshold = float(th)

            if f1 > best_f1:
                best_f1 = float(f1)
                best_f1_threshold = float(th)

            metrics_list.append({
                "threshold": float(th),
                "tar": float(tar),
                "far": float(far),
                "frr": float(frr),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            })

        metrics_df = pd.DataFrame(metrics_list)
        self.results["threshold_metrics"] = metrics_df
        self.results["threshold_summary"] = {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "best_f1": float(best_f1),
            "best_f1_threshold": float(best_f1_threshold)
        }
        logger.info(f"EER: {eer:.4f} at threshold {eer_threshold:.4f} | Best F1: {best_f1:.4f} at threshold {best_f1_threshold:.4f}")

    def generate_plots_and_visualizations(self):
        """Generates publication-quality charts."""
        logger.info("Generating plots and visualizations...")
        sns.set_theme(style="darkgrid")
        
        # 1. Similarity Distributions Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.results["genuine_scores"], label="Genuine Matches", fill=True, color="#22c55e", alpha=0.4)
        sns.kdeplot(self.results["impostor_scores"], label="Impostor Matches", fill=True, color="#ef4444", alpha=0.4)
        plt.axvline(self.results["threshold_summary"]["eer_threshold"], color="#38bdf8", linestyle="--", label=f"EER Thresh ({self.results['threshold_summary']['eer_threshold']:.3f})")
        plt.title("Genuine vs Impostor Cosine Similarity Distributions", fontsize=14, fontweight="bold")
        plt.xlabel("Cosine Similarity", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "similarity_distributions.png"), dpi=300)
        plt.close()

        # 2. Threshold vs Metric Curves (FAR / FRR / TAR / F1)
        tm_df = self.results["threshold_metrics"]
        plt.figure(figsize=(10, 6))
        plt.plot(tm_df["threshold"], tm_df["far"], label="FAR (False Accept Rate)", color="#ef4444", linewidth=2)
        plt.plot(tm_df["threshold"], tm_df["frr"], label="FRR (False Reject Rate)", color="#f59e0b", linewidth=2)
        plt.plot(tm_df["threshold"], tm_df["f1_score"], label="F1 Score", color="#38bdf8", linewidth=2)
        plt.axvline(self.results["threshold_summary"]["eer_threshold"], color="gray", linestyle=":", label="EER Intersect")
        plt.title("Error Rates & F1 Score vs. Cosine Similarity Threshold", fontsize=14, fontweight="bold")
        plt.xlabel("Similarity Threshold", fontsize=12)
        plt.ylabel("Rate", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "threshold_sweeps.png"), dpi=300)
        plt.close()

        # 3. ROC Curve
        plt.figure(figsize=(8, 8))
        plt.plot(tm_df["far"], tm_df["tar"], color="#38bdf8", linewidth=2.5, label="ROC Curve")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
        plt.xlabel("False Accept Rate (FAR)", fontsize=12)
        plt.ylabel("True Accept Rate (TAR)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"), dpi=300)
        plt.close()

        # 4. Identity Confusion Heatmap (Top 15 confused identities)
        conf_pairs = self.results["confusion_pairs"].head(20)
        if not conf_pairs.empty:
            pivot_conf = conf_pairs.pivot(index="query_kerberos", columns="hard_negative_kerberos", values="confusion_count").fillna(0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_conf, annot=True, fmt=".0f", cmap="YlOrRd", cbar=True)
            plt.title("Top Identity Hard Negative Confusion Matrix", fontsize=14, fontweight="bold")
            plt.xlabel("Confused Impostor Kerberos ID", fontsize=12)
            plt.ylabel("Query Kerberos ID", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=300)
            plt.close()

    def generate_diagnostics_and_reports(self):
        """Export raw CSVs, JSON summaries, and Markdown report."""
        logger.info("Exporting summary artifacts, CSVs, and markdown report...")
        
        # Save raw dataframes to CSV
        self.results["sample_results"].to_csv(os.path.join(self.output_dir, "sample_analysis.csv"), index=False)
        self.results["hard_negatives"].to_csv(os.path.join(self.output_dir, "hard_negatives.csv"), index=False)
        self.results["per_identity_stats"].to_csv(os.path.join(self.output_dir, "per_identity_stats.csv"), index=False)
        self.results["within_dp_consistency"].to_csv(os.path.join(self.output_dir, "within_dp_consistency.csv"), index=False)
        self.results["across_dp_consistency"].to_csv(os.path.join(self.output_dir, "across_dp_consistency.csv"), index=False)
        self.results["threshold_metrics"].to_csv(os.path.join(self.output_dir, "threshold_metrics.csv"), index=False)

        # Failure Cases / Diagnostics
        negative_margins = self.results["sample_results"][self.results["sample_results"]["margin"] < 0]
        weak_genuine = self.results["sample_results"].sort_values(by="worst_genuine_sim").head(15)
        strong_impostor = self.results["hard_negatives"].sort_values(by="impostor_similarity", ascending=False).head(15)

        # General summary dictionary
        summary_dict = {
            "dataset_summary": {
                "total_samples": len(self.metadata),
                "total_identities": int(len(self.results["per_identity_stats"])),
                "total_data_points": int(self.results["sample_results"].groupby("kerberos_id")["data_point_number"].nunique().sum()),
            },
            "similarity_stats": {
                "genuine": {
                    "mean": float(np.mean(self.results["genuine_scores"])),
                    "median": float(np.median(self.results["genuine_scores"])),
                    "std": float(np.std(self.results["genuine_scores"])),
                    "min": float(np.min(self.results["genuine_scores"])),
                    "max": float(np.max(self.results["genuine_scores"])),
                },
                "impostor": {
                    "mean": float(np.mean(self.results["impostor_scores"])),
                    "median": float(np.median(self.results["impostor_scores"])),
                    "std": float(np.std(self.results["impostor_scores"])),
                    "min": float(np.min(self.results["impostor_scores"])),
                    "max": float(np.max(self.results["impostor_scores"])),
                }
            },
            "retrieval_metrics": self.results["retrieval_metrics"],
            "threshold_summary": self.results["threshold_summary"],
            "failure_cases_count": {
                "negative_margin_samples": len(negative_margins)
            }
        }

        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary_dict, f, indent=4)

        # Markdown Report Generation
        report_md = f"""# Face Recognition Embedding Analysis Report

## Executive Summary
- **Total Samples Analyzed**: {summary_dict['dataset_summary']['total_samples']}
- **Total Identities (Students)**: {summary_dict['dataset_summary']['total_identities']}
- **Total Data Points**: {summary_dict['dataset_summary']['total_data_points']}
- **Optimal Threshold (EER Intersect)**: `{summary_dict['threshold_summary']['eer_threshold']:.4f}` (Equal Error Rate: `{summary_dict['threshold_summary']['eer']:.2%}`)
- **Max F1-Score Threshold**: `{summary_dict['threshold_summary']['best_f1_threshold']:.4f}` (F1 Score: `{summary_dict['threshold_summary']['best_f1']:.4f}`)

---

## 1. Similarity Distribution Statistics

| Match Type | Count | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|---|
| **Genuine** | {len(self.results['genuine_scores']):,} | {summary_dict['similarity_stats']['genuine']['mean']:.4f} | {summary_dict['similarity_stats']['genuine']['median']:.4f} | {summary_dict['similarity_stats']['genuine']['std']:.4f} | {summary_dict['similarity_stats']['genuine']['min']:.4f} | {summary_dict['similarity_stats']['genuine']['max']:.4f} |
| **Impostor** | {len(self.results['impostor_scores']):,} | {summary_dict['similarity_stats']['impostor']['mean']:.4f} | {summary_dict['similarity_stats']['impostor']['median']:.4f} | {summary_dict['similarity_stats']['impostor']['std']:.4f} | {summary_dict['similarity_stats']['impostor']['min']:.4f} | {summary_dict['similarity_stats']['impostor']['max']:.4f} |

---

## 2. Top-K Retrieval Performance

- **Top-1 Accuracy**: `{summary_dict['retrieval_metrics']['top_1_accuracy']:.2%}`
- **Top-3 Accuracy**: `{summary_dict['retrieval_metrics']['top_3_accuracy']:.2%}`
- **Top-5 Accuracy**: `{summary_dict['retrieval_metrics']['top_5_accuracy']:.2%}`
- **Top-10 Accuracy**: `{summary_dict['retrieval_metrics']['top_10_accuracy']:.2%}`

---

## 3. Data Point Consistency Analysis

- **Within-Data-Point Mean Similarity**: `{self.results['within_dp_consistency']['mean_sim'].mean():.4f}` (Std: `{self.results['within_dp_consistency']['mean_sim'].std():.4f}`)
- **Across-Data-Point Mean Similarity**: `{self.results['across_dp_consistency']['mean_sim'].mean():.4f}` (Std: `{self.results['across_dp_consistency']['mean_sim'].std():.4f}`)

---

## 4. Hardest and Easiest Identities

### Easiest Identities (Highest Margin)
{self.results['per_identity_stats'].head(5)[['kerberos_id', 'avg_genuine_margin', 'avg_genuine_sim', 'highest_impostor_sim']].to_markdown(index=False)}

### Hardest Identities (Lowest Margin)
{self.results['per_identity_stats'].tail(5)[['kerberos_id', 'avg_genuine_margin', 'avg_genuine_sim', 'highest_impostor_sim']].to_markdown(index=False)}

---

## 5. Failure Diagnostics & Negative Margin Samples

- **Total Samples with Negative Margin (Impostor closer than Genuine)**: `{len(negative_margins)}`

### Top Hardest Impostor Matches
{strong_impostor[['query_kerberos', 'query_dp', 'query_sample', 'hard_negative_kerberos', 'impostor_similarity']].head(8).to_markdown(index=False)}

---

## 6. Generated Visualizations
- `similarity_distributions.png`: Density plots of genuine vs. impostor scores.
- `threshold_sweeps.png`: FAR, FRR, and F1 score curves across thresholds.
- `roc_curve.png`: Receiver Operating Characteristic curve.
- `confusion_matrix.png`: Top hard-negative identity confusion matrix.
"""

        with open(os.path.join(self.output_dir, "analysis_report.md"), "w") as f:
            f.write(report_md)

        logger.info(f"Analysis complete! All artifacts exported to: {self.output_dir}")

    def run_pipeline(self):
        """Executes full analysis pipeline sequentially."""
        self.load_dataset()
        self.compute_pairwise_similarity()
        self.analyze_similarities_and_margins()
        self.analyze_data_point_consistency()
        self.analyze_per_identity_and_confusion()
        self.evaluate_thresholds()
        self.generate_plots_and_visualizations()
        self.generate_diagnostics_and_reports()


if __name__ == "__main__":
    analyzer = EmbeddingAnalyzer()
    analyzer.run_pipeline()
