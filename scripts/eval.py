"""
Evaluation script for HarmonyAI recommendation system.

This script:
- Loads test queries with ground truth
- Uses TF-IDF index and mood model to rank songs
- Computes Precision@K and nDCG@K metrics
- Saves results to CSV report
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mood_model import HarmonyMoodModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_precision_at_k(
    relevant_items: List[str],
    retrieved_items: List[str],
    k: int
) -> float:
    """
    Compute Precision@K.

    Args:
        relevant_items: List of relevant item IDs
        retrieved_items: List of retrieved item IDs (in rank order)
        k: Cutoff position

    Returns:
        Precision@K score
    """
    if k <= 0 or not retrieved_items:
        return 0.0

    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)

    num_relevant_in_top_k = sum(1 for item in top_k if item in relevant_set)

    return num_relevant_in_top_k / k


def compute_dcg_at_k(
    relevant_items: List[str],
    retrieved_items: List[str],
    k: int
) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    Args:
        relevant_items: List of relevant item IDs
        retrieved_items: List of retrieved item IDs (in rank order)
        k: Cutoff position

    Returns:
        DCG@K score
    """
    if k <= 0 or not retrieved_items:
        return 0.0

    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_set:
            # Binary relevance: 1 if relevant, 0 otherwise
            # DCG formula: sum(rel_i / log2(i+2))
            dcg += 1.0 / np.log2(i + 2)

    return dcg


def compute_ndcg_at_k(
    relevant_items: List[str],
    retrieved_items: List[str],
    k: int
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        relevant_items: List of relevant item IDs
        retrieved_items: List of retrieved item IDs (in rank order)
        k: Cutoff position

    Returns:
        nDCG@K score
    """
    dcg = compute_dcg_at_k(relevant_items, retrieved_items, k)

    # Ideal DCG: all relevant items at top positions
    ideal_retrieved = relevant_items[:k]
    idcg = compute_dcg_at_k(relevant_items, ideal_retrieved, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_queries(
    queries_path: str,
    dataset_path: str,
    index_dir: str,
    reports_dir: str
) -> None:
    """
    Evaluate test queries and compute metrics.

    Args:
        queries_path: Path to test queries JSON file
        dataset_path: Path to cleaned dataset CSV
        index_dir: Directory containing TF-IDF index
        reports_dir: Directory to save evaluation reports
    """
    logger.info("=" * 60)
    logger.info("Evaluating Queries")
    logger.info("=" * 60)

    # Load test queries
    logger.info(f"Loading test queries from {queries_path}")
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} test queries")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} songs")

    # Load TF-IDF index
    index_path = Path(index_dir)
    vectorizer_path = index_path / 'tfidf_vectorizer.joblib'
    matrix_path = index_path / 'song_tfidf.npz'

    logger.info(f"Loading TF-IDF vectorizer from {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    logger.info(f"Loading TF-IDF matrix from {matrix_path}")
    tfidf_matrix = load_npz(matrix_path)

    # Initialize mood model
    mood_axes = ['valence', 'energy', 'tension']
    mood_model = HarmonyMoodModel(mood_axes)

    # Evaluate each query
    results = []

    for i, query_obj in enumerate(queries):
        query_text = query_obj['query']
        relevant_artists = query_obj['relevant_artists']

        logger.info(f"\nQuery {i+1}/{len(queries)}: '{query_text}'")
        logger.info(f"Relevant artists: {relevant_artists}")

        # Convert query to TF-IDF vector
        query_vector = vectorizer.transform([query_text])

        # Compute cosine similarity with all songs
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Rank songs by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_artists = df.iloc[ranked_indices]['artist'].tolist()

        # Compute metrics
        precision_5 = compute_precision_at_k(relevant_artists, ranked_artists, 5)
        precision_10 = compute_precision_at_k(relevant_artists, ranked_artists, 10)
        ndcg_5 = compute_ndcg_at_k(relevant_artists, ranked_artists, 5)
        ndcg_10 = compute_ndcg_at_k(relevant_artists, ranked_artists, 10)

        logger.info(f"  Precision@5:  {precision_5:.3f}")
        logger.info(f"  Precision@10: {precision_10:.3f}")
        logger.info(f"  nDCG@5:       {ndcg_5:.3f}")
        logger.info(f"  nDCG@10:      {ndcg_10:.3f}")

        results.append({
            'query': query_text,
            'relevant_artists': ', '.join(relevant_artists),
            'precision_at_5': precision_5,
            'precision_at_10': precision_10,
            'ndcg_at_5': ndcg_5,
            'ndcg_at_10': ndcg_10
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Compute averages
    avg_metrics = {
        'query': 'AVERAGE',
        'relevant_artists': '',
        'precision_at_5': results_df['precision_at_5'].mean(),
        'precision_at_10': results_df['precision_at_10'].mean(),
        'ndcg_at_5': results_df['ndcg_at_5'].mean(),
        'ndcg_at_10': results_df['ndcg_at_10'].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)

    # Save to CSV
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    output_path = reports_path / 'metrics.csv'

    logger.info(f"\nSaving results to {output_path}")
    results_df.to_csv(output_path, index=False)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    print()
    print(results_df.to_string(index=False))
    print()
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate HarmonyAI recommendation system'
    )
    parser.add_argument(
        '--queries',
        default='tests/test_queries.json',
        help='Path to test queries JSON (default: tests/test_queries.json)'
    )
    parser.add_argument(
        '--dataset',
        default='data/processed/songs_clean.csv',
        help='Path to cleaned dataset (default: data/processed/songs_clean.csv)'
    )
    parser.add_argument(
        '--index_dir',
        default='data/index',
        help='Directory containing TF-IDF index (default: data/index)'
    )
    parser.add_argument(
        '--reports_dir',
        default='reports',
        help='Directory to save reports (default: reports)'
    )

    args = parser.parse_args()

    try:
        evaluate_queries(args.queries, args.dataset, args.index_dir, args.reports_dir)
        logger.info("\nSUCCESS")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
