from pathlib import Path

BASE = Path(__file__).parent

REDDIT_DATASET = BASE / "reddit/reddit_user_data_count.csv"
REDDIT_META = BASE / "reddit/subreddit_info.csv"

# TODO LoL dataset path

from .downloader import check_all_datasets
