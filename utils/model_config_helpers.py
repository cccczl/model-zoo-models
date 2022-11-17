#!/usr/bin/env python3

import json
import os
import shlex
import subprocess

dry_run = False


def dataset_is_downloaded(dataset_id, dataset_variant):
    marker = f"workspace/datasets/{dataset_id}/{dataset_variant}/.zoo_download_finished"

    return bool(os.path.exists(marker))


def model_is_downloaded(model_id, model_variant):
    marker = f"workspace/models/{model_id}/{model_variant}/.zoo_download_finished"
    return bool(os.path.exists(marker))


def cmdListToString(cmdList):
    command = [shlex.quote(x) for x in cmdList]
    return ' '.join(command)


def logCmd(args, skip=False):
    # show what we are running to the terminal
    line = cmdListToString(args)
    print(line)

    if skip:
        return

    if not dry_run:
        subprocess.check_call(args)


def run_model_test():
    config = json.load(open('model_config.json'))
    # print(config)

    if config.get('dataset'):
        if config['dataset']['train_variant']:
            dataset_id = config['dataset']['dataset_id']
            variant_id = config['dataset']['train_variant']
            skip = False
            if dataset_is_downloaded(dataset_id, variant_id):
                print(
                    f'Already downloaded dataset {dataset_id} {variant_id}, skipping dataset download.'
                )

                skip = True

            logCmd(["leip", "zoo", "download", "--dataset_id", dataset_id, "--variant_id", variant_id], skip=skip)
        if config['dataset']['eval_variant']:
            dataset_id = config['dataset']['dataset_id']
            variant_id = config['dataset']['eval_variant']
            skip = False
            if dataset_is_downloaded(dataset_id, variant_id):
                print(
                    f'Already downloaded dataset {dataset_id} {variant_id}, skipping dataset download.'
                )

                skip = True

            logCmd(["leip", "zoo", "download", "--dataset_id", dataset_id, "--variant_id", variant_id], skip=skip)

    if config.get('base_model_for_training'):
        model_id = config['base_model_for_training']['model_id']
        variant_id = config['base_model_for_training']['variant_id']
        skip = False
        if model_is_downloaded(model_id, variant_id):
            print(
                f'Already downloaded model {model_id} {variant_id}, skipping model download.'
            )

            skip = True

        logCmd(["leip", "zoo", "download", "--model_id", model_id, "--variant_id", variant_id], skip=skip)

    quick_train_cmd = ['python3', 'train.py'] + shlex.split(config['quick_train_args'])
    logCmd(quick_train_cmd)

    eval_cmd = ['python3', 'eval.py'] + shlex.split(config['evaluate_args'])
    logCmd(eval_cmd)

    demo_cmd = ['python3', 'demo.py'] + shlex.split(config['demo_args'])
    logCmd(demo_cmd)

    print('Model test passed.')
    exit(0)
