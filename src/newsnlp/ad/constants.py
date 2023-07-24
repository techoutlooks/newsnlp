import os
from os.path import dirname as up, realpath

from newsnlp.utils.helpers import get_env_variable


# Storage
# ########################################################


# Customizable data dir
data_dir = get_env_variable(
    'DATA_DIR', f"{up(up(up(up(realpath(__file__)))))}/data")


# Storage for fitted data preparation pipeline
saved_preprocessor = f"{data_dir}/preprocessor.pkl"


# Storage for NN's training params
# ie., weights, optimizer params and training history
saved_pretrained = dict(
    f_params=f"{data_dir}/model.pkl",
    f_optimizer=f"{data_dir}/opt.pkl",
    f_history=f"{data_dir}/history.json"
)


# Training
# ########################################################

#  an <a> tag that embeds an <img> child
AD_LOCATOR = '//img/parent::a'
# Dataset generated from website urls, labeled using our `ad-label-creator` ReactJS app
ADS_DATASET = f"{data_dir}/ads-dataset.json"
# Dataset's 1D stats (counts per categories, etc.) required for model weights calibration
ADS_DATASET_STATS = f"{data_dir}/ads-stats.json"

# Seed for randomizing training data
SEED = 7
MIN_PHRASE_COUNT = 10
# Max num
MAX_PHRASE_LEN = 2
# Median length of phrase's tokens
MIN_TOKEN_LEN = 2


