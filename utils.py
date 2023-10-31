"""
This module contains utility functions
"""


def process_to_dict(data_file, key, append_to={}):
    """
    Takes the .mat data file and extract the data associated 
    the key @key and restructure it in dict format:
        {
            "key" : [LIST OF LISTS]
        }
    """
    data = []
    for el in data_file:
        d = []
        for i in el:
            d.append(round(list(i)[0], 8))
        data.append(d)
    try:
        append_to[key].append(data)
    except:
        append_to[key] = data


def get_correct_key(keys_list, key_start):
    """
    return the right key from @keys_list starting with @key_start
    """
    for key in keys_list:
        if key.startswith(key_start):
            return key
        
def get_max_num_points(data: dict) -> int:
    max_data = []
    for key in data.keys():
        max_data.append(len(max(data[key], key=lambda el: len(el))))

    return max(max_data)

def get_min_num_points(data: dict) -> int:
    max_data = []
    for key in data.keys():
        max_data.append(len(min(data[key], key=lambda el: len(el))))

    return max(max_data)