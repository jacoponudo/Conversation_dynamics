import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/PRO'
sys.path.append(module_path)


def add_percentile_column(group):
    group['percentile'] = (group['created_at'].rank(pct=True) * 100).round(2)
    return group

from tqdm import tqdm

def count_unique_words(text):
    if isinstance(text, str):
        words = text.split()
        return len(set(words))
    else:
        return 0

