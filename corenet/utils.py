from time import sleep
from datetime import datetime
import hashlib

def get_unique_label():
    """Generate unique time from system clock.
    
    This function is NOT thread-safe.

    Returns:
        str: Unique label
    """

    sleep(1e-6)
    now = str(datetime.now()).encode('ascii')
    return hashlib.md5(now).hexdigest()
