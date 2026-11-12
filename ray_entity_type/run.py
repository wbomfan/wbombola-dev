import logging
import sys
import time

import click
import pandas as pd
import ray
from ray.data import DataContext

from inference import EntityTypePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("entity-type-ray")

INPUT_COLUMNS = ["wiki_id", "article_id", "plaintext"]


def clean(batch: pd.DataFrame) -> pd.DataFrame:
    batch = batch.rename(columns={"plaintext": "plaintext_content"}).dropna()
    text = batch["plaintext_content"].astype(str)
    return batch[(text.str.strip() != "") & (text != "nan")]


@click.command()
@click.option("--input_s3_path", required=True)
@click.option("--output_s3_path", required=True)
@click.option("--batch_size", default=128, type=int, show_default=True)
@click.option("--num_actors", default=1, type=int, show_default=True, help="GPU actors; one per available GPU")
@click.option("--num_gpus", default=1.0, type=float, show_default=True, help="GPUs per actor; 0 for CPU runs")
@click.option("--limit", default=None, type=int, help="Process only the first N rows")
@click.option("--min_rows_per_file", default=100_000, type=int, show_default=True)
def main(input_s3_path, output_s3_path, batch_size, num_actors, num_gpus, limit, min_rows_per_file):
    start = time.perf_counter()
    DataContext.get_current().enable_progress_bars = sys.stdout.isatty()
    ray.init()
    log.info("cluster resources: %s", ray.cluster_resources())

    ds = ray.data.read_parquet(input_s3_path, columns=INPUT_COLUMNS)
    if limit:
        ds = ds.limit(limit)

    predictions = ds.map_batches(clean, batch_format="pandas").map_batches(
        EntityTypePredictor,
        batch_size=batch_size,
        batch_format="pandas",
        concurrency=num_actors,
        num_gpus=num_gpus,
    )
    predictions.write_parquet(output_s3_path, min_rows_per_file=min_rows_per_file)

    log.info("wrote to %s in %.0fs", output_s3_path, time.perf_counter() - start)


if __name__ == "__main__":
    main(auto_envvar_prefix="ETR")
