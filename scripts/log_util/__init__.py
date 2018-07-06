import glob
import os
import os.path
from datetime import datetime

def get_log_index(log_dir):
    try:
        return int(os.path.basename(log_dir).split('-')[0])
    except:
        return None
        
def get_training_log_dir(root_dir, log_index, step_index, opt_name = None):
    """Get log directory
    
    Args:
        root_dir (str): Root directory of log.
        log_index (int): Index of log directory. If None, index will solve automatically. 
        step_index (int): Index of learning stage. 
        opt_name (str, optional): Prefix of log directory (Default: '')
    
    Raises:
        ValueError: If log directory is empty when step_index over than 2
    
    Returns:
        str: log directory
    """

    if log_index is None:
        if step_index == 1: # 1st stage training and log index is automatically generation
            return get_new_training_log_dir(root_dir, opt_name=opt_name)
        else:                # After 1st stage training and log directory is user selected.
            return root_dir
    else:
        if step_index == 1: # 1st stage training and log index user defined.
            return get_new_training_log_dir(root_dir, start_index=log_index, opt_name=opt_name)
        else:                # After 1st stage training and log index user defined.
            log_dirs = [ log_dir for log_dir in glob.glob(os.path.join(root_dir, str(log_index) + '-*')) if os.path.isdir(log_dir)]
            if len(log_dirs) == 0:
                raise ValueError('Selected index directory is not found: {}\nVerify the root directory: {}'.format(log_index, root_dir))
            return log_dirs[0]
            

def get_new_training_log_dir(root_dir, opt_name = None, start_index = 0):
    log_dirs = [ log_dir for log_dir in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(log_dir)]
    max_id = -1
    for log_dir in log_dirs:
        log_dir = os.path.basename(log_dir)
        pos = log_dir.find('-')
        if pos == -1:
            continue
        try:
            tmp_max_id = max(max_id, int(log_dir[:pos]))
            if start_index == tmp_max_id:   # Selected index is duplicated so increase index and continue to check duplicating.
                start_index += 1
            max_id = tmp_max_id
        except ValueError:
            pass
    
    if max_id <= start_index: # Found index less than use selected index
        max_id = start_index - 1

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    cur_dir = ''
    if opt_name is not None:
        cur_dir = '{}-{}-TIME-{}'.format(max_id + 1, opt_name, timestamp)
    else:
        cur_dir = '{}-TIME-{}'.format(max_id + 1, timestamp)

    out = os.path.join(root_dir, cur_dir)
    return out