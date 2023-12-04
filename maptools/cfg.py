from utils.misc import detect_os
from utils.memory import create_memory_cache


# set `MEMORY`
OS_NAME = detect_os()
if OS_NAME == "Mac":
    MEMORY = create_memory_cache('/Users/wenke/Documents/Cache', verbose=0)
else:
    MEMORY = create_memory_cache('../Cache', verbose=1)

# set `KEY`
# KEY = "33464a61337565a8b73a32e8de13e6e7"
KEY = "25db7e8486211a33a4fcf5a80c22eaf0" # personal
assert KEY != "", "Config `KEY` first."

# Coordnination System
LL_SYS = 'wgs'