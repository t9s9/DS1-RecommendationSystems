from pathlib import Path

import kaggle
import streamlit

from src.data import REDDIT_DATASET, REDDIT_META

# To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your
# user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the
# download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json
# (on Windows in the location C:\Users\<Windows-username>\.kaggle\kaggle.json - you can check the exact location,
# sans drive, with echo %HOMEPATH%). You can define a shell environment variable KAGGLE_CONFIG_DIR to change this
# location to $KAGGLE_CONFIG_DIR/kaggle.json (on Windows it will be %KAGGLE_CONFIG_DIR%\kaggle.json).

s = """ To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your
user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the
download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json
(on Windows in the location C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json - you can check the exact location,
sans drive, with echo %HOMEPATH%). You can define a shell environment variable KAGGLE_CONFIG_DIR to change this
location to $KAGGLE_CONFIG_DIR/kaggle.json (on Windows it will be %KAGGLE_CONFIG_DIR%\kaggle.json). """


def check_all_datasets():
    check_reddit()


def check_reddit():
    if not (Path(REDDIT_META).exists() and Path(REDDIT_DATASET).exists()):
        try:
            download_location = Path(REDDIT_DATASET).parent
            api = kaggle.KaggleApi()
            api.authenticate()
            api.dataset_download_files("timschaum/subreddit-recommender", unzip=True, path=download_location,
                                       quiet=False)
        except Exception as e:
            streamlit.error(f"Something went wrong with your Kaggle Account. \n\nTo use the automatic python downloader"
                            f" follow these instructions:"
                            f"\n{s}\n\nOtherwise go to https://www.kaggle.com/timschaum/subreddit-recommender and "
                            f"download the two csv files (reddit_user_data_count.csv, subreddit_info.csv) manually and "
                            f"put them into src/data/reddit")
            print(e)


def check_lol():
    # TODO
    pass
