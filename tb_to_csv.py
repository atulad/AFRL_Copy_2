"""Convert tensorboard metric to csv"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CSV_FOLDER_PATH = 'results/csv'
TB_FOLDER_PATH = 'results/runs'

def extract(dpath, metrics):
    scalar_accumulators = [EventAccumulator(os.path.join(dpath, dname)).Reload(
    ).scalars for dname in os.listdir(dpath)]

    def extract_for_key(key):
        trials = [scalar_accumulator.Items(
            key) for scalar_accumulator in scalar_accumulators]

        def flatten(ys): return [x for xs in ys for x in xs]
        results = [[(episode, observation.step, observation.wall_time, observation.value)
                    for (episode, observation) in enumerate(trial)] for trial in trials]
        results = flatten(results)
        results = list(zip(*sorted(results, key=lambda x: x[1])))
        return results

    return {key: extract_for_key(key) for key in metrics}


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '_', s)


def write_csv(dpath, extracts):
    path = os.path.join(CSV_FOLDER_PATH, os.path.basename(dpath))

    if not os.path.exists(path):
        os.makedirs(path)

    def write_csv_key(key, episodes, steps, wall_times, values):
        file_name = get_valid_filename(key) + '.csv'
        df = pd.DataFrame({'episode': episodes, 'step': steps,
                          'wall_time': wall_times, 'value': values})
        df.to_csv(os.path.join(path, file_name), index=None)

    for key, (episodes, steps, wall_times, values) in extracts.items():
        write_csv_key(key, episodes, steps, wall_times, values)


def main(path, metrics):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Path {} does not exist".format(path))

    extracts = extract(path, metrics)
    write_csv(path, extracts)


if __name__ == '__main__':
    metrics = ['afrl/forecast', 'rollout/ep_rew_mean']
    for fname in ['P_AF_SAC_1']:
        path = os.path.join(TB_FOLDER_PATH, fname)
        main(path, metrics)
