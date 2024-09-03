import logging
import mmap
import os

import numpy as np

try:
    import orjson as json
except ImportError:
    import json

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from . import json_functions

def change_extension(path, new_extension):
    path = str(path)
    return path.rpartition(".")[0] + new_extension


def find_newline_positions(path, show_progress=True, leave_progress=True):
    path = str(path)
    indexes = []
    with open(path, "r") as f:
        indexes.append(f.tell())
        pbar = tqdm(
            total=os.path.getsize(path),
            desc="Finding newlines for mmindex",
            unit="B",
            unit_scale=True,
            disable=not show_progress,
            leave=leave_progress,
        )

        while f.readline():
            t = f.tell()
            indexes.append(t)
        
            if pbar is not None:
                pbar.update(t - indexes[-2])
        
        if pbar is not None:
            pbar.close()

    return indexes[:-1]


def save_mmindex(indexes, path):
    path = str(path)
    index_file = change_extension(path, ".mmindex.json")
    with open(index_file, "w") as f:
        f.write(json_functions.dumps(indexes))


def load_mmindex(path):
    path = str(path)
    index_file = change_extension(path, ".mmindex.json")
    with open(index_file, "r") as f:
        return json_functions.loads(f.read())


# now we can jump to any line in the file thanks to the index and mmap
def get_line(
    path,
    index,
    mmindex,
    encoding="utf-8",
    file_obj=None,
    mmap_obj=None,
) -> str:
    path = str(path)
    if file_obj is None:
        file_obj = open(path, "r")
        CLOSE_FILE = True
    else:
        CLOSE_FILE = False

    if mmap_obj is None:
        mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        CLOSE_MMAP = True
    else:
        CLOSE_MMAP = False

    mmap_obj.seek(mmindex[index])
    result = mmap_obj.readline().decode(encoding)

    if CLOSE_MMAP:
        mmap_obj.close()

    if CLOSE_FILE:
        file_obj.close()

    return result


class JsonlCorpus:
    """
    A class to read a jsonl file line by line using mmap, allowing extremely fast
    access to any line in the file. For example, you could access the N-th line
    of a 10GB file in a fraction of a second, returning a dictionary.

    Example
    --------

    Traditioanally, you would read a jsonl file line by line like this:

    ```python
    import json
    data = [json.loads(line) for line in open("file.jsonl")]
    print(corpus[1000])
    ```

    This is memory inefficient and has a large overhead. Instead, you can use this class:

    ```python
    corpus = JsonlCorpus("file.jsonl")
    print(corpus[1000])

    ```

    Which only loads the line you need into memory, and is much faster.
    """

    def __init__(self, path, show_progress=True, leave_progress=True, save_index=True, verbosity=1):
        self.path = path
        self.verbosity = verbosity

        # if the index file does not exist, create it
        if os.path.exists(change_extension(path, ".mmindex.json")):
            self.mmindex = load_mmindex(path)
        else:
            logging.info("Creating index file for jsonl corpus")
            mmindex = find_newline_positions(
                path, show_progress=show_progress, leave_progress=leave_progress
            )
            if save_index:
                save_mmindex(mmindex, path)

            self.mmindex = mmindex

        # Finally, open the file and mmap objects
        self.load()

    def __len__(self):
        return len(self.mmindex)

    def __getitem__(self, index):
        # handle multiple indices
        if isinstance(index, int):
            return json_functions.loads(
                get_line(
                    self.path,
                    index,
                    self.mmindex,
                    file_obj=self.file_obj,
                    mmap_obj=self.mmap_obj,
                )
            )

        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        if isinstance(index, (list, tuple)):
            return [self.__getitem__(i) for i in index]
        if isinstance(index, np.ndarray):
            # if it's an ndarray, this means each element is an index, and the array can
            # be of any shape. thus, we should flatten it first, get the results as if it
            # was a list, and then reshape it back to the original shape
            index_flat = index.flatten().tolist()
            results = [self.__getitem__(i) for i in index_flat]
            reshaped = np.array(results).reshape(index.shape)
            return reshaped
        
        raise TypeError("Invalid index type")

    def close(self):
        """
        Close the file and mmap objects. This is useful if you want to free up memory. To reopen them, use the `load` method.
        If you don't call this method, the objects will be closed automatically when the object is deleted.
        """
        if hasattr(self, "file_obj") and self.file_obj is not None:
            self.file_obj.close()
            # delete the object
            del self.file_obj
            self.file_obj = None
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            self.mmap_obj.close()
            # delete the object
            del self.mmap_obj
            self.mmap_obj = None
        if self.verbosity >= 1:
            logging.info("Closed file and mmap objects")
    
    def load(self):
        """
        Load the file and mmap objects. This is useful if you closed them and want to reopen them.

        Note
        ----
        This is called automatically when the object is created. You don't need to call it manually.
        Also, if there is an existing file and mmap object, this will close them before reopening.
        """
        self.close()  # close any existing file and mmap objects

        self.file_obj = open(self.path, "r")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        if self.verbosity >= 1:
            logging.info("Opened file and mmap objects")
    
    def __del__(self):
        self.close()
