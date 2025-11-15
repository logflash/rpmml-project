import os
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    def save(self):
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None):
        args = super().parse_args(known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.get_commit(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        self.save_diff(args)
        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        """
        Override config parameters with command-line arguments.
        Accepts BOTH:
            key=value
        and
            --key value
        """

        extras = args.extra_args
        if not len(extras):
            return

        print(f"[ utils/setup ] Found extras: {extras}")

        # Remove experiment name if present ("diffusion", "values", etc.)
        if extras[0] in ["diffusion", "values", "plan"]:
            extras = extras[1:]

        i = 0
        while i < len(extras):
            item = extras[i]

            # --- Format: key=value ---
            if "=" in item and not item.startswith("--"):
                key, val = item.split("=", 1)

            # --- Format: --key value ---
            elif item.startswith("--"):
                key = item.replace("--", "")
                # safe: if no value follows, break
                if i + 1 < len(extras):
                    val = extras[i + 1]
                    i += 1
                else:
                    print(f"[ utils/setup ] WARNING: flag {item} has no value, skipping")
                    break

            else:
                print(f"[ utils/setup ] WARNING: unrecognized arg format '{item}', skipping")
                i += 1
                continue

            # --- Apply override ---
            assert hasattr(args, key), f"[ utils/setup ] {key} not found in config {args.config}"

            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f"[ utils/setup ] Overriding config | {key}: {old_val} --> {val}")

            # Convert type
            if val == 'None':
                val = None
            elif old_type is bool:
                val = val.lower() in ["true", "1", "yes"]
            else:
                try:
                    val = old_type(val)
                except:
                    print(f"[ utils/setup ] WARNING: failed to convert '{val}' to {old_type}, keeping as string")

            setattr(args, key, val)
            self._dict[key] = val

            i += 1


    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None:
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')
