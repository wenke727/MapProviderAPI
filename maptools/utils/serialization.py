import time
import hashlib
import pickle
import os


def load_checkpoint(ckpt_file_name, obj=None):
    _dict = {}
    if obj is not None and hasattr(obj, "__dict__"):
        _dict = obj.__dict__

    with open(ckpt_file_name, 'rb') as f:
        dict_ = pickle.load(f)
    _dict.update(dict_)
    
    return _dict
    

def save_checkpoint(obj, ckpt_file_name, ignore_att=[]):
    def _save(tmp):
        with open(ckpt_file_name, 'wb') as f:
            pickle.dump({ k: v for k, v in tmp.items() if k not in ignore_att}, f)
    
    if isinstance(obj, dict):
         _save(obj)
         return True
     
    try:                
        _save(obj.__dict__)
        return True
    except:
        return False