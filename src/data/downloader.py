import kaggle
from src.data import REDDIT_DATASET, REDDIT_META
from pathlib import Path


# To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your
# user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the
# download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json
# (on Windows in the location C:\Users\<Windows-username>\.kaggle\kaggle.json - you can check the exact location,
# sans drive, with echo %HOMEPATH%). You can define a shell environment variable KAGGLE_CONFIG_DIR to change this
# location to $KAGGLE_CONFIG_DIR/kaggle.json (on Windows it will be %KAGGLE_CONFIG_DIR%\kaggle.json).

def check_all_datasets():
    check_reddit()


def check_reddit():
    if not (Path(REDDIT_META).exists() and Path(REDDIT_DATASET).exists()):
        download_location = Path(REDDIT_DATASET).parent
        api = kaggle.KaggleApi()
        api.authenticate()
        api.dataset_download_files("timschaum/subreddit-recommender", unzip=True, path=download_location, quiet=False)


def check_lol():
    # TODO
    pass
