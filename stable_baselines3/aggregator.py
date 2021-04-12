# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

FOLDER_NAME = 'aggregates'


def extract(dpath):
    scalar_accumulators = [EventAccumulator(str(dpath / dname)).Reload(
    ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME]
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]

    def extract_for_key(key):
        trials = [scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators]
        flatten = lambda ys: [x for xs in ys for x in xs]
        results = [[(episode, observation.step, observation.wall_time, observation.value) for (episode, observation) in enumerate(trial)] for trial in trials]
        results = flatten(results)
        results = list(zip(*sorted(results, key=lambda x: x[1])))
        return results

    return {key: extract_for_key(key) for key in ['rollout/ep_rew_mean']}


def write_summary(dpath, extracts):
    dpath = dpath / FOLDER_NAME / dpath.name
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, values) in extracts.items():
        for episode, step, wall_time, value in zip(steps, wall_times, values):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
            scalar_event = Event(episode=episode, wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '_', s)


def write_csv(dpath, extracts):
    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    def write_csv_key(key, episodes, steps, wall_times, values):
        file_name = get_valid_filename(key) + '.csv'
        df = pd.DataFrame({'episode': episodes, 'step': steps, 'wall_time': wall_times, 'value': values})
        df.to_csv(path / file_name, index=None)

    for key, (episodes, steps, wall_times, values) in extracts.items():
        write_csv_key(key, episodes, steps, wall_times, values)


def aggregate(dpath, output):
    name = dpath.name

    ops = {
        'summary': write_summary,
        'csv': write_csv
    }

    print("Started aggregation {}".format(name))

    extracts = extract(dpath)

    ops[output](dpath, extracts)

    print("Ended aggregation {}".format(name))


def main(path):
    path = Path(path)
    output = 'csv'
    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    subpaths = [path / dname for dname in os.listdir(path) if dname != FOLDER_NAME]

    for subpath in subpaths:
        if not os.path.exists(subpath):
            raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    if output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(output))

    aggregate(path, output)

if __name__ == '__main__':
    basepath = 'runs/'
    for fname in ['MC_AF_SAC_2']: #['MC_SAC', 'MC_AF_SAC', 'P_SAC', 'P_AF_SAC', 'CP_DQN', 'CP_AF_DQN', 'LL_SAC', 'LL_AF_SAC_n=10', 'LL_AF_SAC_n=24']:
        main(basepath + fname)
