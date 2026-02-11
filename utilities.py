"""
A collection of utility functions for data handling, file I/O, and common operations.
Author: Takumi Matsuzawa
"""
import numpy as np
import re
import h5py
import pickle
import os
import copy
import pandas as pd
from pathlib import Path

# READING AND WRITING FILES
# HDF5
def isListOfStrings(obj):
    """
    Check if a list is a list of strings

    Parameters
    ----------
    obj: object

    Returns
    -------
    bool
    """
    if isinstance(obj, list):
        return all(isinstance(item, str) for item in obj)
    else:
        return False

def isListOfArrays(obj):
    """
    Check if a list is a list of arrays

    Parameters
    ----------
    obj: object

    Returns
    -------
    bool
    """
    if isinstance(obj, list):
        return all(isinstance(item, np.ndarray) for item in obj)
    else:
        return False

def isListOfLists(obj):
    """
    Check if a list is a list of lists

    Parameters
    ----------
    obj: object

    Returns
    -------
    bool
    """
    if isinstance(obj, list):
        return all(isinstance(item, list) for item in obj)
    else:
        return False

# READ / WRITE
# HDF5

# h5 helpers
def show_h5_contents(dpath):
    """
    Displays the contents of an HDF5 file in a tree-like structure.

    Parameters
    ----------
    dpath: str, path to the HDF5 file

    Returns
    -------
    None
    """
    def print_name(name, obj):
        indent = "  " * (name.count('/'))
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- {name.split('/')[-1]} (Dataset, shape={obj.shape})")
        else:
            print(f"{indent}+ {name.split('/')[-1]} (Group)")
    if os.path.exists(dpath):
        with h5py.File(dpath, "r") as f:
            f.visititems(print_name)
    else:
        print(f'{dpath} does not exist.')

def show_h5_keys(dpath):
    """Displays a list of dataset names in h5- only the datasets stored under the top"""
    with h5py.File(dpath, mode='r') as f:
        print(f.keys())


def show_h5_subkeys(dpath, key):
    """Displays a list of dataset names in h5- only the datasets stored under the top"""
    with h5py.File(dpath, mode='r') as f:
        try:
            print(f[key].keys())
        except KeyError:
            print(f'{key} does not exist in {dpath}')


def get_h5_keys(dpath):
    """Returns a list of dataset names in h5- only the datasets stored under the top
    ... /Dataset1, /Dataset2, ... -> Returns ["Dataset1", "Dataset2"]
    """
    with h5py.File(dpath, mode='r') as f:
        keys = [key for key in f.keys()]
    return keys


def get_h5_subkeys(dpath, key):
    """Returns a list of dataset names in h5- only the datasets stored under /key
    ... key/Dataset1, key/Dataset2, ... -> Returns ["Dataset1", "Dataset2"]
    """
    with h5py.File(dpath, mode='r') as f:
        subkeys = [subkey for subkey in f[key].keys()]
    return subkeys

def write_h5(filename, datadict, attrdict=None, overwrite=False, verbose=True):
    """
    Saves a nested dictionary in an HDF5 format
    ... Supports strings, lists of strings, lists of lists, and lists of arrays

    Parameters
    ----------
    filepath: str
        ... file path where data will be stored. (Do not include extension- .h5)
    datadict: dict, a collection of data to be stored in the h5 file
        ... data_dict = {'varname0': var0, 'varname1': var1, ...}
        ... data will be stored in the h5 file under the group /varname0, /varname1, ...
            and can be accessed by
            ... with h5py.File(filepath, mode='r') as f:
                    data = f[key][:]
    attrdict: dict, default: None, metadata of the data
        ... Attributes of data to be stored in the h5 file
        ... Organize the attributes of a variable like this:
            attrdict = {'varname0': {'attr0': value0, 'attr1': value1, ...}, ...}
            ... attrdict = {'piv000': {'int_win': 32, 'date': 01012020, 'software': 'DaVis'},
                            'piv001': {'int_win': 16, 'date': 01012020, 'software': 'PIVLab'},
                            ...}
    bool: overwrite, default: False
    verbose: bool, default: True
        ... print where the data is stored

    Returns
    -------
    None
    """

    # Helpers
    def isListOfStrings(x):
        return isinstance(x, (list, tuple)) and all(isinstance(i, str) for i in x)

    def isListOfLists(x):
        return isinstance(x, (list, tuple)) and all(
            isinstance(i, (list, np.ndarray)) and not isinstance(i, str) for i in x)

    def isListOfArrays(x):
        return isinstance(x, (list, tuple)) and all(isinstance(i, np.ndarray) for i in x)

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    if not filename.endswith('.h5'):
        filename += '.h5'

    def _write_group(h5obj, data_dict, attr_dict=None):
        # Group-level attributes or nested metadata groups
        if attr_dict:
            # Split keys into metadata for children vs group-only
            keys_in_data = set(data_dict.keys())
            for a, v in attr_dict.items():
                if a in keys_in_data:
                    # Per-dataset (handled later)
                    continue
                # Nested dict metadata => subgroup
                if isinstance(v, dict):
                    subgrp = h5obj.require_group(a)
                    if verbose:
                        print(f"Using subgroup '{subgrp.name}' for nested metadata")
                    _write_group(subgrp, {}, v)
                else:
                    # Primitive metadata => group attribute
                    h5obj.attrs[a] = v
                    if verbose:
                        print(f"Attached group attr '{a}' to '{h5obj.name}'")
        # Now handle child datasets/groups
        for key, value in data_dict.items():
            child_attr = attr_dict.get(key) if attr_dict else None
            if isinstance(value, dict):
                grp = h5obj.require_group(key)
                if verbose:
                    print(f"Using group '{grp.name}'")
                _write_group(grp, value, child_attr)
            else:
                # Prepare data
                if isinstance(value, str):
                    data = np.string_(value)
                elif isListOfStrings(value):
                    maxlen = max(len(item) for item in value)
                    data = np.array(value, dtype=f'S{maxlen}')
                elif isListOfLists(value) or isListOfArrays(value):
                    arrays = [np.asarray(arr, float) for arr in value]
                    ml = max(arr.shape[-1] for arr in arrays)
                    padded = [np.pad(arr, (0, ml - arr.shape[-1]), constant_values=np.nan) for arr in arrays]
                    data = np.stack(padded, axis=0)
                else:
                    data = value
                # Overwrite
                if key in h5obj and overwrite:
                    del h5obj[key]
                if key in h5obj:
                    if verbose:
                        print(f"Dataset '{h5obj[key].name}' exists; skipping")
                else:
                    ds = h5obj.create_dataset(key, data=data)
                    if verbose:
                        print(f"Created dataset '{ds.name}' shape {ds.shape}")
                    # Attach per-dataset attrs
                    if child_attr:
                        for an, av in child_attr.items():
                            ds.attrs[an] = av
                            if verbose:
                                print(f"  - Attached attr '{an}' to '{ds.name}'")

    with h5py.File(filename, 'a') as hf:
        _write_group(hf, datadict, attrdict or {})

    if verbose:
        print(f"Data saved to {filename}")


def read_simple_h5(datapath, grpname=None):
    """
    Returns a dictionary of data in a hdf5
    1. ASSUMED DATA ARCHITECTURE (DEFAULT)
    ... The hdf5 file must have a simple structure.
        ... /DataA
        ... /DataB
        ... /DataC
    ... The returned dictionary has a structure like following.
        datadit['DataA'] = values of DataA
        datadit['DataB'] = values of DataB
    2. OPTIONAL ARCHITECTURE
    ... It is possible to read a COLLECTION of simple data
        ... /grp0/DataA
        ... /grp0//DataB
        ... /grp0//DataC
        ... /grp1/DataA
        ... /grp1//DataB
        ... /grp1//DataC

    Parameters
    ----------
    datapath: str, path to the hdf5 file
    grpname: str, default: None
        ... 'grpname' is the name of the group in the h5 nameto be read.
    Returns
    -------
    datadict: dict
        ... datadict['DataA'] = values of /DataA
    """
    datadict = {}
    with h5py.File(datapath, 'r') as data:
        if grpname is not None:
            data = data[grpname]
        keys = list(data.keys())
        for key in keys:
            datadict[key] = data[key][...]
        print('Keys of the returning dictionary: ', keys)
    return datadict

# LEGACY: DO NOT USE
# def read_data_h5(h5path, keys, return_dict=False, grpname=None, verbose=True):
#     """
#     Grabs data in a simply organized h5 file
#     ... Return the data stored at /keys[0], /keys[1], ...
#
#     Parameters
#     ----------
#     h5path: str, a path to a h5 file
#     keys: list, a list of keys
#         if
#
#     Returns
#     -------
#     data_read: list of data stroed in a h5 file- [data0, data1, data2, ...]
#     or
#     datadict: dictionary of data- {name0: data0, name1: data1, name2: data2, ...}
#     """
#     if h5path[-3:] != '.h5':
#         h5path += '.h5'
#     if keys == 'all':
#         keys = get_h5_keys(h5path)
#     if not return_dict:
#         data_read = []
#         with h5py.File(h5path, mode='r') as f:
#             for key in keys:
#                 try:
#                     if grpname is None:
#                         val = f[key][...]
#                     elif grpname in f.keys():
#                         val = f[grpname][key][...]
#                     else:
#                         if verbose:
#                             print('... /%s does not exist in the given h5' % grpname)
#                         sys.exit()
#                     data_read.append(val)
#                 except:
#                     if verbose:
#                         print('read_data_from_h5: %s does not exist in %s' % (key, h5path))
#                     data_read.append(None)
#         return data_read
#     else:
#         datadict = {}
#         with h5py.File(h5path, mode='r') as f:
#             for key in keys:
#                 try:
#                     if grpname is None:
#                         val = f[key][...]
#                     elif grpname in f.keys():
#                         val = f[grpname][key][...]
#                     else:
#                         if verbose:
#                             print('... /%s does not exist in the given h5' % grpname)
#                         sys.exit()
#                     datadict[key] = val
#                 except:
#                     if verbose:
#                         print('read_data_from_h5: %s does not exist in %s' % (key, h5path))
#                     datadict[key] = None
#         return datadict

def read_data_h5(
    filepath: str,
    keys,
    return_dict: bool = False,
    recursive: bool = True,
    grpname: str = None,
    verbose: bool = True
):
    """
    Reads datasets or subgroups from an HDF5 file by full path or relative to a group (backward-compatible).

    # Example usage:
    # 1) Flatten-only (default) returns a dict of immediate datasets under the group:
    #    read_h5_data("mydata.h5", ["/GroupA/SubGroupA1"], return_dict=True)
    #
    # 2) Full nested subtree:
    #    read_h5_data("mydata.h5", ["/GroupA/SubGroupA1"], return_dict=True, recursive=True)

    Parameters
    ----------
    filepath : str
        Path to the .h5 file (extension optional).
    keys : list of str or 'all'
        If grpname is None, full HDF5 paths (e.g. '/A/A1/key').
        If grpname is provided, keys are names under that group (relative): ['ds1', 'sub/ds2'] or 'all'.
    as_dict : bool, default False
        If True, returns a dict {path_or_name: data}. Otherwise returns a list of data.
    recursive : bool, default False
        If True and a key refers to a group, reads the entire subtree into a nested dict.
        If False and the key is a group, reads only its immediate datasets into a flat dict.
    grpname : str, default None
        Backward-compatible: if provided, all keys are looked up under this group.
    verbose : bool, default True
        If True, prints warnings for missing paths or type mismatches.

    Returns
    -------
    list or dict
        The data read from the file (arrays or dicts).
    """
    def _read_group_tree(group):
        tree = {}
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                tree[name] = item[...]
            else:
                tree[name] = _read_group_tree(item)
        return tree
    if isinstance(filepath, Path):
        filepath = str(filepath)

    if not filepath.endswith('.h5'):
        filepath += '.h5'

    if isinstance(filepath, Path):
        filepath = str(filepath)

    with h5py.File(filepath, 'r') as f:
        # Determine base object: root or subgroup for backward compatibility
        base = f
        if grpname:
            if grpname in f:
                base = f[grpname]
            else:
                if verbose:
                    print(f"Group '{grpname}' not found; using root.")
                base = f

        # Expand 'all' under base group
        if keys == 'all':
            keys_to_process = list(base.keys())
        else:
            keys_to_process = list(keys)

        result = {} if return_dict else []

        for key in keys_to_process:
            # If using full paths and no grpname, split absolute path
            if grpname is None and key.startswith('/'):
                parts = [p for p in key.strip('/').split('/') if p]
                obj = f
                for p in parts:
                    if isinstance(obj, h5py.Group) and p in obj:
                        obj = obj[p]
                    else:
                        if verbose:
                            print(f"Path '{key}' not found in {filepath}.")
                        obj = None
                        break
            else:
                # Relative to base group
                parts = [p for p in key.strip('/').split('/') if p]
                obj = base
                for p in parts:
                    if isinstance(obj, h5py.Group) and p in obj:
                        obj = obj[p]
                    else:
                        if verbose:
                            print(f"Path '{key}' not found under '{grpname or '/'}'.")
                        obj = None
                        break

            # Determine read behavior
            if obj is None:
                data = None
            elif isinstance(obj, h5py.Dataset):
                data = obj[...]
            elif isinstance(obj, h5py.Group):
                if recursive:
                    data = _read_group_tree(obj)
                else:
                    data = {name: ds[...] for name, ds in obj.items() if isinstance(ds, h5py.Dataset)}
            else:
                data = None

            if return_dict:
                result[key] = data
            else:
                result.append(data)

    return result

def _remove_datasets_in_group(group):
    """Recursively delete all Dataset objects under this h5py.Group."""
    for name in list(group.keys()):
        obj = group[name]
        if isinstance(obj, h5py.Dataset):
            del group[name]
        elif isinstance(obj, h5py.Group):
            _remove_datasets_in_group(obj)

def remove_data_from_h5(h5_path, target_paths, recursive=False, verbose=True):
    """
    Remove one or more datasets (or group contents) in an HDF5 file.

    # Example usage:
    # 1. Remove multiple datasets:
    # remove_h5_paths("susceptibility_data.h5",
    #                 ["/Bik1/60uM/AcETF/fluoro/c_solute",
    #                  "/Bik1/60uM/AcEETF/fluoro/c_Bik1"],
    #                 recursive=False)

    # 2. Clear out all datasets under several groups:
    # remove_h5_paths("susceptibility_data.h5",
    #                 ["/Bik1/60uM/AcETF/fluoro",
    #                  "/Bik1/60uM/AcEETF/fluoro"],
    #                 recursive=True)
    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.
    target_paths : str or list of str
        Full HDF5 path(s) to the object(s) you want to remove, e.g.
        "/Bik1/…/c_solute" or ["/grp/ds1", "/grp/subgrp"].
    recursive : bool, default False
        If True and a target path is a group, delete all datasets under it.
        If False and a target is a group, raises an error.
        Ignored for dataset paths.
    """
    # normalize to a list
    if isinstance(target_paths, str):
        target_paths = [target_paths]

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "a") as f:
        for tgt in target_paths:
            if tgt not in f:
                print(f"Path '{tgt}' not found in file.")
                continue

            obj = f[tgt]
            parent_path, name = os.path.split(tgt.rstrip("/"))
            parent = f[parent_path] if parent_path else f["/"]

            if isinstance(obj, h5py.Dataset):
                # delete the single dataset
                del parent[name]
                if verbose:
                   print(f"Removed {tgt} (dataset)'.")

            elif isinstance(obj, h5py.Group):
                if recursive:
                    ## Option 1: Keep a group but delete all datasets inside it
                    ## delete all datasets under this group
                    # _remove_datasets_in_group(obj)
                    ## Option 2: Remove the group entirely
                    del parent[name]
                    if verbose:
                       print(f"Removed {tgt} (group)'.")
                else:
                    raise ValueError(
                        f"'{tgt}' is a group. "
                        "Set recursive=True to delete its contents."
                    )

            else:
                raise TypeError(f"Unknown object type at '{tgt}'.")

        f.flush()

def add_data_h5(dpath, datadict, attrdict=None, grpname=None, overwrite=False, verbose=True,
                restricted_keys=['ux', 'uy', 'uz', 'x', 'y', 'z'],
                ):
    """
    Writes a data stored in a dictionary into a hdf5 at udatatapth
    ... datadict = {"name1": value1, "name2":value2, ...} will be stored like /name1, /name2 in the hdf5 file
    ... datadict = {"grp1": {"name1": value1, "name2":value2, ...},
                    "grp1": {"name1": value1, "name2":value2, ...}, ... }
        will be stored like /grp1/name1, /grp1/name2, /grp2/name1, /grp2/name2, ...

    Parameters
    ----------
    dpath: str, a path to the hdf5 file
    datadict: dictionary, or a nested dictionary (up to level 2)
        ... data must be stored like {"name1": value1, "name2": value2, ...}
    grpname: str
        ... if given, it saves the datadict like
            /grpname/name1, /grpname/name2
    overwrite: bool, if True, it overwrite the data in the target hdf5 file
    verbose: bool, if True, it prints out details during saving the data

    Returns
    -------
    None
    """
    def format_datadict(datadict):

        """
        Format the data to be saved in the hdf5 file
        Parameters
        ----------
        datadict: dict, a dictionary to be saved in the hdf5 file

        Returns
        -------
        datadict: dict, a formatted dictionary
        """
        for key in datadict.keys():
            # Check if the data is a list of strings
            # If so, convert it to a numpy array of strings
            if isListOfStrings(datadict[key]):
                maxLength = max([len(item) for item in datadict[key]])
                datadict[key] = np.array(datadict[key], dtype=f'S{maxLength}')
            elif isListOfLists(datadict[key]) or isListOfArrays(datadict[key]):
                arrays = datadict[key]
                # Find the maximum length of arrays
                max_length = max(len(arr) for arr in arrays)
                # Create a padded array
                padded_array = np.array(
                    [np.pad(np.array(arr, dtype=float), (0, max_length - len(np.array(arr, dtype=float))), constant_values=np.nan) for arr in arrays])

                datadict[key] = padded_array
        return datadict
    datadict = format_datadict(datadict)

    if grpname is not None:
        if verbose:
            print('add_data_h5: data will be saved like /%s/names' % grpname)
            print('... the given datadict must not be a nested dictionary')
        datadict = {grpname: datadict}
    # 
    # restricted_keys = ['ux', 'uy', 'uz', 'x', 'y', 'z']  # these keys will be ignored to protect original data

    # check if datadict contains data that must not be overwritten
    new_keys = [key for key in datadict.keys() if key not in restricted_keys]
    for key in datadict.keys():
        if key in restricted_keys:
            print(key, ' is forbidden to overwrite. To write udata to a h5 file, use write_udata()')

    if not os.path.exists(os.path.split(dpath)[0]):
        os.makedirs(os.path.split(dpath)[0])
    with h5py.File(dpath, mode='a') as f:
        existing_keys = f.keys()
        for new_key in new_keys:
            if overwrite and type(datadict[new_key]) != dict:
                try:
                    if verbose:
                        print('add_data_h5: Adding %s...' % new_key)
                    f.create_dataset(new_key, data=datadict[new_key])
                except:
                    del f[new_key]
                    f.create_dataset(new_key, data=datadict[new_key])
                    if verbose:
                        print('add_data_h5: %s already exists. Overwriting...' % new_key)
            else:
                if new_key in existing_keys and type(datadict[new_key]) != dict:
                    if verbose:
                        print('add_data_h5: %s already exists! Skipping...' % new_key)
                else:
                    if type(datadict[new_key]) == dict: # {'key1': {'subkey1': data}}
                        if not new_key in existing_keys:
                            grp = f.create_group('/%s/' % new_key)
                        else:
                            grp = f[new_key]
                        subkeys = datadict[new_key].keys()
                        existing_subkeys = f[new_key].keys()
                        for subkey in subkeys:
                            if type(datadict[new_key][subkey]) == dict:
                                subsubkeys = datadict[new_key][subkey].keys()
                                if subkey not in existing_subkeys:
                                    subgrp = f.create_group('/%s/%s' % (new_key, subkey))
                                else:
                                    subgrp = f[new_key][subkey]
                                for subsubkey in subsubkeys:
                                    try:
                                        subgrp.create_dataset(subsubkey, data=datadict[new_key][subkey][subsubkey])
                                        if verbose:
                                            print('add_data_h5: Adding /%s/%s/%s...' % (
                                                new_key, subkey, subsubkey))
                                    except:
                                        if overwrite:
                                            del grp[subkey][subsubkey]
                                            subgrp.create_dataset(subsubkey, data=datadict[new_key][subkey][subsubkey])
                                            print('add_data_h5: /%s/%s/% already exists. Overwriting...' % (
                                                new_key, subkey, subsubkey))
                                        else:
                                            if verbose:
                                                print('add_data_h5: /%s/%s/%s already exists! Skipping...' % (
                                                    new_key, subkey, subsubkey))
                            else:
                                try:
                                    grp.create_dataset(subkey, data=datadict[new_key][subkey])
                                    if verbose:
                                        print('add_data_h5: Adding /%s/%s...' % (
                                            new_key, subkey))
                                except:
                                    if overwrite:
                                        del grp[subkey]
                                        grp.create_dataset(subkey, data=datadict[new_key][subkey])
                                        print('add_data_h5: /%s/%s already exists. Overwriting...' % (
                                            new_key, subkey))
                                    else:
                                        if verbose:
                                            print('add_data_h5: /%s/%s already exists! Skipping...' % (
                                                new_key, subkey))
                    else:
                        f.create_dataset(new_key, data=datadict[new_key])

        if attrdict is not None:
            for key in attrdict:
                for attrname, item in attrdict[key].items():
                    if attrname not in f[key].attrs or overwrite:
                        print(f'add_data_h5: {key}.attrs[\'{attrname}\'] already exists. Overwriting...')
                        f[key].attrs[attrname] = item
                    else:
                        print(f'add_data_h5: {key}.attrs[\'{attrname}\'] already exists. Skipping...')

# PICKLE
def write_pickle(filepath, obj, verbose=True):
    """
    Generate a pickle file from obj

    Parameters
    ----------
    obj: object
    filepath: str, path to the pickle file
    verbose: bool, default: True

    Returns
    -------
    None
    """
    # Extract the directory and filename from the given path
    directory, filename = os.path.split(filepath)[0], os.path.split(filepath)[1]
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle_out = open(filepath, "wb")
    pickle.dump(obj, pickle_out)
    if verbose:
        print('Saved data as ' + filepath)
    pickle_out.close()

def read_pickle(filename):
    """
    Read a pickle file

    Parameters
    ----------
    filename: str, path to the pickle file

    Returns
    -------
    obj: object, the object stored in the pickle file
    """
    with open(filename, "rb" ) as pickle_in:
        try:
            obj = pickle.load(pickle_in)
        except UnicodeDecodeError:
            try:
                obj = pickle.load(pickle_in, encoding="bytes")
            except:
                obj = pd.read_pickle(filename)
    return obj





# FIND NEAREST VALUE
def find_nearest(array, value, option='auto'):
    """
    Find an element and its index closest to 'value' in 'array'

    Parameters
    ----------
    array: nd array
    value: float/int
    option: str, 'auto' or 'less', or 'greater'
        ... 'auto' will return the element closest to 'value'
        ... 'less' will return the element closest to 'value' that is LESS than 'value'
        ... 'greater' will return the element closest to 'value' that is GREATER than 'value'

    Returns
    -------
    idx: index of the array where the closest value to 'value' is stored in 'array'
    array[idx]: the closest value in 'array'
    """
    # get the nearest value such that the element in the array is LESS than the specified 'value'
    if option == 'less':
        array_new = copy.copy(array)
        array_new[array_new > value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    # get the nearest value such that the element in the array is GREATER than the specified 'value'
    if option == 'greater':
        array_new = copy.copy(array)
        array_new[array_new < value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    else:
        array_new = copy.copy(array)
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array[idx]

# NATURAL SORTING
def natural_sort(arr):
    """
    natural-sorts elements in a given array
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    e.g.-  arr = ['a28', 'a01', 'a100', 'a5']
    ... WITHOUT natural sorting,
     -> ['a01', 'a100', 'a28', 'a5']
    ... WITH natural sorting,
     -> ['a01', 'a5', 'a28', 'a100']


    Parameters
    ----------
    arr: list or numpy array of strings

    Returns
    -------
    sorted_array: natural-sorted
    """

    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    return sorted(arr, key=natural_keys)


# SORTING MULTIPLE ARRAYS
def sortNArrays(list_of_arrays, element_dtype=np.ndarray):
    """
    Sort a list of N arrays by the order of the first array in the list,
    ensuring that np.nan values appear at the end.
    e.g. a=[2, 1, np.nan, 3], b=[1, 9, np.nan, 8], c=['a', 'b', np.nan, 'c']
        [a, b, c] -> [(1, 2, 3, np.nan), (9, 1, 8, np.nan), ('b', 'a', 'c', np.nan)]

    Parameters
    ----------
    list_of_arrays: list of lists/1D arrays
        A list containing N lists or 1D arrays to be sorted.
    element_dtype: data type, default: tuple
        The data type of the elements in the returned list.
        Possible values are tuple, list, or np.ndarray.

    Returns
    -------
    list_of_sorted_arrays: list of sorted lists/1D arrays
        The sorted list of arrays, with the specified element data type.
    """

    def is_nan(value):
        """Check if the value is NaN."""
        try:
            return np.isnan(value)
        except TypeError:
            return False

    # Combine the arrays into a list of tuples, with np.nan handled specially
    combined = list(zip(*list_of_arrays))

    # Sort the combined list, treating np.nan as larger than any other value
    combined_sorted = sorted(combined, key=lambda x: (is_nan(x[0]), x[0] if not is_nan(x[0]) else float('inf')))

    # Separate the sorted tuples back into individual arrays
    list_of_sorted_arrays = list(zip(*combined_sorted))

    # Convert the sorted arrays to the desired data type
    if element_dtype == list:
        list_of_sorted_arrays = [list(a) for a in list_of_sorted_arrays]
    elif element_dtype == np.ndarray:
        list_of_sorted_arrays = [np.asarray(a) for a in list_of_sorted_arrays]

    return list_of_sorted_arrays

# STRING WRANGLER (PATHS)
def get_filename_wo_ext(filename):
    filename_no_ext, ext = os.path.splitext(filename)
    filename_short = os.path.split(filename_no_ext)
    return filename_short[1]


def get_parent_dir(filename):
    return os.path.dirname(filename)

def get_parent_dir_of_current_dir(dirname):
    return os.path.dirname(os.path.dirname(dirname))

# STRING WRANGLER
def convert_decimalstr_to_float(str):
    """
    Convert a decimal string into float.
    E.g. "5p3" -> 5,3
    decimalstr is, for example, '5p3'. This method translates this string into 5.3 (float).

    Parameters
    ----------
    str: decimal str

    Returns
    -------
    number: float
    """
    return float(str.replace('p', '.'))

def convert_float_to_decimalstr(number, zpad=3):
    """
    Convert float into decimal string which replaces a decimal point with a letter p.
    e.g. 5.3 -> "5p3"

    Parameters
    ----------
    number: decimal str

    Returns
    -------
    str: string
    """
    number_str = str(number)
    number_str = number_str.replace('.', 'p')
    return number_str.rjust(zpad, '0')

def extract_float(str, start, end):
    """
    Extract a number from a string like '_lalala_start5p3end_lalala' -> 5.3 (float)
    Automatically convert to float if decimalstr (ex. 2p6 for 2.6) is given

    Parameters
    ----------
    str: str
    start: str
    end: str

    Returns
    -------
    number: float

    """
    if str.find(start) < 0 and start != '':
        print('ERROR: ' + start + ' is not in ' + str)
        return None
        # raise NameError('issues with ' + start)
        #sys.exit(1)

    if str.find(end) < 0 and end != '':
        print('ERROR: ' + end + ' is not in ' + str)
        return None
        # raise NameError('issues with ' + end)
        #sys.exit(1)
    if start == '':
        start_ind = 0
    else:
        start_ind = str.find(start)

    new_str = str[start_ind + len(start):]
    if end == '':
        end_ind = len(new_str)
    else:
        end_ind = new_str.find(end)
    # str_extracted = str[start_ind+len(start):end_ind]
    str_extracted = new_str[:end_ind]
    try:
        number = float(str_extracted)
        return number
    except ValueError:
        try:
            number = convert_decimalstr_to_float(str_extracted)
            return number
        except ValueError:
            print('ERROR: ' + str_extracted + ' cannot be converted into float')
            raise RuntimeError

def extract_str(str, start, end):
    """
    Extract a string from a string like '_lalala_start5p3end_lalala' -> 5.3 (float)
    Parameters
    ----------
    str:
    start:
    end:

    Returns
    -------
    number: float
    """
    if str.find(start) < 0:
        print('ERROR: ' + start + ' is not in ' + str)
        raise NameError('issues with ' + start)
    if str.find(end) < 0:
        print('ERROR: ' + end + ' is not in ' + str)
        raise NameError('issues with ' + end)
    start_ind, end_ind = str.find(start), str.find(end)
    str_extracted = str[start_ind+len(start):end_ind]
    return str_extracted



def to_sci_notation(number, hideone=True):
    """
    Returns a string in a scientific notation (latex), base 10
    Parameters
    ----------
    number: float ... e.g. 52481
    hideone: bool ... If hideone is True and the result becomes "1 x 10^m", return only "10^m".

    Returns
    -------
    float_str: string ... e.g. "5 x 10^4"

    """
    float_str = "{0:.2g}".format(number)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if int(round(float(base))) == 1 and hideone:
            return r"$10^{{{0}}}$".format(int(exponent))
        else:
            return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))

    else:
        return float_str