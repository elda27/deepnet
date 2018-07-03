import sys
import os
import os.path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import pytest
import log_util
import tempfile
import glob
import warnings
from contextlib import ExitStack

def is_duplicating(root_dir):
    found_dirs = glob.glob(os.path.join(root_dir, '*'))
    index_list = []
    for found_dir in found_dirs:
        log_index = None
        try:
            log_index = int(found_dir.split('-')[0])
        except:
            continue
        if log_index in index_list:
            return True
        index_list.append(log_index)
    return False

@pytest.mark.parametrize(('index', ), [
    (1, ),
    (21, ),
    (90, ),
    (100, ),
])
def test_get_dir(index):
    log_dir = log_util.get_training_log_dir('.', index, 1)
    log_index = os.path.basename(log_dir).split('-')[0]
    assert int(log_index) == index

@pytest.mark.parametrize(('index', ), [
    (1, ),
    (21, ),
    (90, ),
    (100, ),
])
def test_auto_detect_next_index(index):
    with ExitStack() as e_stack:
        created_dirs = []
        def makedir(dirname):
            dirname = os.path.basename(dirname)
            dir_obj = tempfile.TemporaryDirectory(prefix=dirname)
            e_stack.push(dir_obj)
            created_dirs.append(dir_obj.name)

        temp_dir = tempfile.gettempdir()
        for i in range(index):
            new_dir = log_util.get_training_log_dir(temp_dir, i, 1)
            makedir(new_dir)
            assert not is_duplicating(temp_dir)

        log_dir = log_util.get_training_log_dir(temp_dir, None, 1)
        log_index = log_util.get_log_index(log_dir)
        makedir(log_dir)
        assert not is_duplicating(temp_dir)
        assert log_index is not None and log_index >= index
        if log_index != index:
            warnings.warn("Log index is not duplicating so this test maybe fail. log_index != index({} != {})".format(log_index, index))


if __name__ == '__main__':
    pytest.main()
