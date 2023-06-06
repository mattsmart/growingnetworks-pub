import h5py
import numpy as np
import os

from utils_io import run_subdir_setup
from settings import HDF5_COMPRESS, HDF5_COMPRESS_LVL

"""
See Official HDF5 module "h5py" guide here:
https://docs.h5py.org/en/stable/quick.html

An HDF5 file is a container for two kinds of objects:
    - datasets, which are array-like collections of data
    - groups, which are folder-like containers that hold datasets / groups.

Fundamentals of h5py:
    - Groups work like dictionaries, and
    - Datasets work like NumPy arrays

Extras:
    - the hdf5 file is itself a Group
"""


class HDF5Autoclose:
    """
    Small & Simple helper class for HDF5Handler() -- ensures hard close of HDF5 file whenever it is accessed.

    Usage:

    with HDF5Autoclose(file, mode) as f:
        # DO STUFF,  # e.g. f.create_dataset(hdf5_path_to_dataset, data=arr)

    Once outside of the "with:" statement it will "hard close" the file.

    Args:
        - fpath (str):   path to the HDF5 file itself
        - mode (str):    char in ('r', 'w', 'a') -- read/write/append mode
    """
    def __init__(self, fpath, mode):
        self.fpath = fpath
        self.mode = mode

    def __enter__(self):
        self.file_object = h5py.File(self.fpath, self.mode)
        return self.file_object

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_object.close()
        if exc_type is not None:
            print(exc_type, exc_value)
        return


class HDF5Handler():
    """
    Usage:
        - create instance (new file or read one):
            HDF5Helper("path_to_my_hdf5_file")
        - get an overview of the file structure:
            self.overview()
        - unpack into txt files mimicking the hierarchical structure
            self.unpack()

    Args:
        fpath (str):     path to the HDF5 file itself
        io_dict (dict):  "parent" dictionary of paths relevant for a generic simulation/run (e.g. of CellGraph)
        verbose (bool):  controls amount of prints during class behaviour and method calls

    """

    def __init__(self, fpath, io_dict=None, verbose=True):
        self.fpath = fpath
        self.io_dict = io_dict  # if specified: k,v pairs are stored in f.attrs
        self.verbose = verbose  # stored in f.attrs

        # regulations
        assert fpath[-3:] == '.h5'  # let's avoid people doing hdf5

        # 1) check if file already exists (try to load that class instance if so)
        if os.path.exists(fpath):
            self.vprint("init() -- note an .h5 file already exists at %s" % fpath)
        else:
            self.vprint("init() -- fresh .h5 will be initialized at %s" % fpath)

        with HDF5Autoclose(self.fpath, 'a') as f:
            # manage root attributes
            f.attrs['verbose'] = verbose
            if self.io_dict:
                f.attrs.update(self.io_dict)

    def vprint(self, msg):
        if self.verbose:
            print('v0_hdf5handler:', msg)

    def overview(self):
        """
        Prints name of each group and dataset by walking over the file
        Note: the visit() method requires a callable argument
        """
        def h5_iter_printname(name, obj):
            is_dataset = isinstance(obj, h5py.Dataset)
            if is_dataset:
                smid = 'Dataset: %s %s' % (obj.dtype, obj.shape)
            else:
                smid = 'Group'
            print('\t%s |' % name, smid, '|', dict(obj.attrs))

        print('\n%s\nHDF file overview: %s\n%s' % ('=' * 50, self.fpath, '=' * 50))
        print('\thpath=%s' % self.fpath)
        print('\tverbose=%s' % self.verbose)
        self.read_attributes('/')
        print('File structure:')
        with HDF5Autoclose(self.fpath, 'r') as f:
            f.visititems(h5_iter_printname)
        print('end overview\n')
        return

    def read_attributes(self, hpath):
        """
        Prints name of each group and dataset by walking over the file
        Note: the visit() method requires a callable argument
        """
        print('Attributes for %s, hpath=%s' % (self.fpath, hpath))
        with HDF5Autoclose(self.fpath, 'r') as f:
            internal = f[hpath]
            list_attrs = list(internal.attrs.items())
            print('(n=%d) attrs found...' % len(list_attrs))
            for pair in list_attrs:
                print('\t', pair)
        return

    '''
    def ORIG_write_dataset(self, hpath_to_store, arr, attrs={}):
        """
        hpath_to_store (str): use format "/grp1/sub_grp/here"
        Notes:
            - arr must be a numpy arr for the usage style of f.create_dataset(*, data=NUMPY_ARR)
            - else, assumes (and checks) that arr is a list of strings (e.g. CellGraph.labels)
        """
        with HDF5Autoclose(self.fpath, 'a') as f:

            if isinstance(arr, np.ndarray):
                dset = f.create_dataset(hpath_to_store, data=arr)

            else:
                assert isinstance(arr, list)
                assert isinstance(arr[0], str)
                arr = np.array(arr).astype('S')
                dset = f.create_dataset(hpath_to_store, data=arr)

            self.vprint('f.write_dataset(), store arr size %s (%s) at %s' % (arr.shape, arr.dtype, hpath_to_store))
            # append attributes if any are passed
            dset.attrs.update(attrs)
        return dset
    '''

    def write_dataset(self, hpath_to_store, arr, rm=False, attrs={}):
        """
        hpath_to_store (str): use format "/grp1/sub_grp/here"
        rm (bool): if True, delete the dataset with the specified name in order to re-write it
        Notes:
            - arr must be a numpy arr for the usage style of f.create_dataset(*, data=NUMPY_ARR)
            - else, assumes (and checks) that arr is a list of strings (e.g. CellGraph.labels)
            - compression is possible using kwargs: https://docs.h5py.org/en/stable/high/dataset.html
        """
        with HDF5Autoclose(self.fpath, 'a') as f:
            if rm:
                if hpath_to_store in f:
                    del f[hpath_to_store]
            dset = f.create_dataset(hpath_to_store,
                                    data=arr,
                                    compression=HDF5_COMPRESS,
                                    compression_opts=HDF5_COMPRESS_LVL)
            self.vprint('f.write_dataset(), store arr size %s (%s) at %s' % (arr.shape, arr.dtype, hpath_to_store))
            dset.attrs.update(attrs)  # append attributes if any are passed
        return dset

    def read_dataset(self, hpath, read_attr=True, inspect=False):
        """
        hpath (str): use format "/data/is/here/"
        Returns the hdf5 data at that location as a np array (of the same size and type)
        """
        with HDF5Autoclose(self.fpath, 'r') as f:
            dset = f[hpath]
            arr = np.array(dset)
            if inspect:
                print('Inspection of %s, hpath=%s' % (self.fpath, hpath))
                print('\t type(dset):', type(dset))
                print('\t dset.dtype:', dset.dtype)
                print('\t dset.shape:', dset.shape)
            if read_attr:
                self.read_attributes(hpath)

        return arr

    def unpack(self, dir_unpack=None):
        """
        Purpose: convert hdf5 to list of text files (or npz), similar to what we had originally
        """

        # take first two chars of dtype and map to np.savetxt format arg
        dtype_to_fmt = {
            '|S': "%s",
            'fl': "%.4f",
            'in': "%d",
        }

        # 1) specify "states" sub-directory of the run folder
        if dir_unpack is None:
            if self.io_dict:
                dir_base = self.io_dict['dir_base']
            else:
                dir_base = os.path.split(self.fpath)[0]
            #dir_unpack = dir_base
            dir_unpack = os.path.join(dir_base, "unpack")

        # 2) create dir_states if it does not exist
        if not os.path.exists(dir_unpack):
            os.mkdir(dir_unpack)

        def h5_iter_unpack(name, obj):
            is_dataset = isinstance(obj, h5py.Dataset)
            if is_dataset:
                # CASE: object is Dataset -- create text file
                arr = np.array(obj)
                fmt = dtype_to_fmt[str(arr.dtype)[0:2]]
                # for arrays of strings, force unicode (i.e. remove the b'*' representation)
                if fmt == '%s':
                    arr = np.array([x.decode() for x in arr])
                outpath = os.path.join(dir_unpack, name) + '.txt'
                np.savetxt(outpath, arr, fmt=fmt)

            else:
                # CASE: object is Group -- create folder
                dir_subgrp = os.path.join(dir_unpack, name)
                if not os.path.exists(dir_subgrp):
                    os.mkdir(dir_subgrp)

        # 3) perform unpack, mimicking the hdf5 file structure
        with HDF5Autoclose(self.fpath, 'r') as f:
            f.visititems(h5_iter_unpack)
            self.vprint('Done unpacking hdf5 file to %s' % dir_unpack)

        return None


if __name__ == '__main__':
    print('%s\nExample usage of HDF5Handler() class\n%s' % ('='*50, '='*50))

    example_read = False
    example_generate = True

    if example_read:
        hdf5_to_read = 'input' + os.sep + 'classdata.h5'
        h5_handler = HDF5Handler(hdf5_to_read, verbose=True)
        h5_handler.overview()
        print('Demonstration of unpacking feature...')
        h5_handler.unpack(dir_unpack=os.path.join(os.path.split(hdf5_to_read)[0], 'states_unpack'))

    if example_generate:
        io_dict = run_subdir_setup(run_subfolder='testing_hdf5', minimal_mode=True)
        fpath = io_dict['dir_base'] + os.sep + 'h5py_usage_main.h5'
        h5_handler = HDF5Handler(fpath, io_dict=io_dict, verbose=True)

        h5_handler.overview()
        h5_handler.read_attributes('/')

        # Example: storing then reading a numpy array
        dataset_goes_here = '/grp1/sub_grp/here'
        sample_arr = np.random.rand(7, 5).astype(np.float64)
        dset_written = h5_handler.write_dataset(dataset_goes_here, sample_arr)
        dset_read = h5_handler.read_dataset(dataset_goes_here, inspect=True)
        print('type(dset_read)', type(dset_read), dset_read.dtype)
        h5_handler.overview()

        for idx in range(10):
            hpath_to_store = '/grp2/%d' % idx
            sample_arr = np.random.rand(7, 5).astype(np.float32)
            dset_written = h5_handler.write_dataset(hpath_to_store, sample_arr)
        h5_handler.overview()

        # unpacking the hdf5 file
        unpack_manual = os.path.join(os.path.split(fpath)[0], 'manual_unpack_location')
        h5_handler.unpack(dir_unpack=unpack_manual)
        h5_handler.unpack(dir_unpack=None)
