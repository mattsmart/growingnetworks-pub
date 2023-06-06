import natsort
import numpy as np
import os
import shutil

from class_sweep_cellgraph import SweepCellGraph
from settings import SWEEP_VARIETY_VALID
from utils_io import pickle_load


def assemble_many_2d_sweeps_to_3d(src_dir, p3_name, p3_variety, p3_count, tgt_dir, build_mode='default', srcnested=True):
    """
    E.g. assemble_many_2d_sweeps_to_3d('src', 'alpha_sharing', 'meta_cellgraph', 61, 'src/sweep3d/')

    Assumptions on the directory src_dir:
    - contains exactly K folders and no files
    - these represent "regularly spaced" 2D sweeps of equal size I x J
    - their names indicate what we are sweeping over but that isn't essential
    - each contains a sweep.pkl, storing its 2D sweep class object

    if srcnested:
        - each contains J folders; each of those contains I folders;
            - form j/i/stuff
            - i, j start at 0 and increase
            - stuff contains:
                cellgraph.h5
                cellgraph.pkl
                run_info.txt
    else:
        - each contains I * J folders (in one directory) with the form i_j/stuff
        - i, j start at 0 and increase
        - stuff contains:
            cellgraph.h5
            cellgraph.pkl
            run_info.txt

    build_mode in ['default', 'copy', 'move']
        - default: do not move or copy the src directories, simply build the new sweep.pkl file in tgt_dir
        - copy: the assembly process will copy src directories, renaming the individual cellgraph runs accordingly.
                else, the assembly process will move/rename the individual runs within src_dir
        - move: as above, but move instead of copy (be careful)
        - note regarding "srcnested" argument used with copy/move:
            if src_nested: p2/p1 becomes p3/p2/p1 (three nested levels)
                     else: p1_p2 becomes p1_p2_p3 (all in one directory)

    """
    assert build_mode in ['default', 'copy', 'move']
    assert build_mode != 'move'  # careful
    assert not os.path.exists(tgt_dir)

    def read_value_p3_name_from_runinfo(run_info_fpath):
        valid_param = False
        with open(run_info_fpath, 'r') as f:
            for line in f.read().splitlines():
                pair = line.split(',')
                assert len(pair) == 2
                if pair[0] == p3_name:
                    p3_val = float(pair[1])
                    valid_param = True
                    break
            if not valid_param:
                print('Error, p3_name not found in src_dir/subdir/rundir/run_info.txt for sampled cellgraph run')
                p3_val = None
        return p3_val

    def validate_subdir_and_get_p3_value(subdir_path, sweep_2d_template):
        axis_0_sz = len(sweep_2d_template.params_values[0])
        axis_1_sz = len(sweep_2d_template.params_values[1])

        if srcnested:
            files_and_rundirs = os.listdir(subdir_path)
            files = [f for f in files_and_rundirs if os.path.isfile(subdir_path + os.sep + f) and f[0] != '.']
            rundirs = [f for f in files_and_rundirs if os.path.isdir(subdir_path + os.sep + f) and f[0] != '.']
            rundirs = natsort.realsorted(rundirs)

            assert len(rundirs) == axis_1_sz
            assert len(files) == 1 and files[0] == 'sweep.pkl'
            assert rundirs[0] == '0'
            assert rundirs[-1] == '%d' % (axis_1_sz - 1)

            p3_val = read_value_p3_name_from_runinfo(subdir_path + os.sep + '0' + os.sep + '0' + os.sep + 'run_info.txt')

        else:
            files_and_rundirs = os.listdir(subdir_path)
            files = [f for f in files_and_rundirs if os.path.isfile(subdir_path + os.sep + f) and f[0] != '.']
            rundirs = [f for f in files_and_rundirs if os.path.isdir(subdir_path + os.sep + f) and f[0] != '.']
            rundirs = natsort.realsorted(rundirs)

            assert len(rundirs) == axis_0_sz * axis_1_sz
            assert len(files) == 1 and files[0] == 'sweep.pkl'
            assert rundirs[0] == '0_0'
            assert rundirs[-1] == '%d_%d' % (axis_0_sz - 1, axis_1_sz - 1)

            p3_val = read_value_p3_name_from_runinfo(subdir_path + os.sep + rundirs[0] + os.sep + 'run_info.txt')

        return p3_val

    def build_mode_cases(subdir_sweep_2d_idx, key2d, key3d):
        if build_mode != 'default':
            # B) copy/move relevant file directories
            if srcnested:
                src_subdir = subdir_sweep_2d_idx + os.sep + str(key2d[1]) + os.sep + str(key2d[0])
                tgt_subdir = tgt_dir + os.sep + str(key3d[2]) + os.sep + str(key3d[1]) + os.sep + str(key3d[0])
            else:
                src_subdir = subdir_sweep_2d_idx + os.sep + '%d_%d' % key2d
                tgt_subdir = tgt_dir + os.sep + '%d_%d_%d' % key3d

            if build_mode == 'move':
                shutil.move(src_subdir, tgt_subdir)
            else:
                shutil.copytree(src_subdir, tgt_subdir)
        return

    # 0) basic asserts
    assert p3_variety in SWEEP_VARIETY_VALID
    assert p3_count > 0

    # 1) make sure src_dir is as expected
    src_files_and_subdirs = os.listdir(src_dir)
    src_files = [f for f in src_files_and_subdirs if os.path.isfile(src_dir + os.sep + f) and f[0] != '.']
    src_subdirs = [f for f in src_files_and_subdirs if os.path.isdir(src_dir + os.sep + f) and f[0] != '.']
    assert len(src_subdirs) == p3_count
    assert len(src_files) == 0
    # sort src_subdirs using natsort.realsort (we will check later that this sorting is correct)
    src_subdirs = natsort.realsorted(src_subdirs)

    # 2) load basic info from the first accessible 2d sweep
    subdir_template = src_dir + os.sep + src_subdirs[0]
    fpath_template = subdir_template + os.sep + 'sweep.pkl'
    sweep_2d_template = pickle_load(fpath_template)
    # asserts
    assert p3_name not in sweep_2d_template.params_name
    src_subdirs_runs = [f for f in os.listdir(subdir_template) if os.path.isdir(subdir_template + os.sep + f) and f[0] != '.']
    if srcnested:
        rundir_template = subdir_template + os.sep + '0' + os.sep + '0' + os.sep + 'run_info.txt'
    else:
        rundir_template = subdir_template + os.sep + src_subdirs_runs[0] + os.sep + 'run_info.txt'
    read_value_p3_name_from_runinfo(rundir_template)

    # 3) loop over all 2d sweeps:
    # - verify integrity (minimal info is there, and it matches the template from first sweep)
    # - assemble into 3d sweep (fill in the blanks)
    p3_values = np.zeros(p3_count)
    results_dict_sweep_3d = {}
    for idx, dirname_sweep_2d in enumerate(src_subdirs):
        p3_val = validate_subdir_and_get_p3_value(src_dir + os.sep + dirname_sweep_2d, sweep_2d_template)
        p3_values[idx] = p3_val
        # below: check p3 steps are linear (and check sorting is correct)
        if idx > 1:
            p3_vals_step = p3_values[1] - p3_values[0]
            assert p3_values[0] + p3_vals_step * idx - p3_val <= 1e-9
        # load the idx-associated 2d sweep now
        subdir_sweep_2d_idx = src_dir + os.sep + src_subdirs[idx]
        sweep_2d_idx = pickle_load(subdir_sweep_2d_idx + os.sep + 'sweep.pkl')
        assert sweep_2d_idx.sizes == sweep_2d_template.sizes

        for key2d in np.ndindex(*sweep_2d_template.sizes):
            key3d = (key2d[0], key2d[1], idx)
            # A) extract result dict object
            results_dict_sweep_3d[key3d] = sweep_2d_idx.results_dict[key2d]
            # B) copy/move relevant file directories
            build_mode_cases(subdir_sweep_2d_idx, key2d, key3d)

    # 4) make skeleton of 3d sweep, then fill it in
    sweep_label_3d = os.path.basename(tgt_dir)
    params_name_3d = sweep_2d_template.params_name + [p3_name]
    params_variety_3d = sweep_2d_template.params_variety + [p3_variety]
    params_values_3d = sweep_2d_template.params_values + [p3_values]

    sweep_3d = SweepCellGraph(
        sweep_label=sweep_label_3d,
        base_cellgraph_kwargs=sweep_2d_template.base_kwargs,
        params_name=params_name_3d,
        params_values=params_values_3d,
        params_variety=params_variety_3d,
        solver_kwargs=sweep_2d_template.solver_kwargs,
        dir_root=os.path.dirname(tgt_dir))
    sweep_3d.results_dict = results_dict_sweep_3d
    sweep_3d.pickle_save()

    print('Merge 2D sweeps to 3D complete')

    return sweep_3d


if __name__ == '__main__':
    build_mode = 'default'  # must be in ['default', 'copy', 'move']
    srcnested = False

    src_dir = "runs" + os.sep + "nested_blob_multi_sweep2d"
    tgt_dir = "runs" + os.sep + "nested_merged_sweep3d_from_blob_multi_sweep2d"

    num_runs = 61
    assemble_many_2d_sweeps_to_3d(src_dir, "alpha_sharing", "meta_cellgraph", num_runs, tgt_dir,
                                  build_mode=build_mode, srcnested=srcnested)
