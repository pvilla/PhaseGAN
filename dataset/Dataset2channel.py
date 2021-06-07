import h5py
import torch
from pathlib import Path
from torch.utils import data

class Dataset2channel(data.Dataset):
    """Dataloader for h5 files. It is based on hdf5 dataset by B. Holländer"""
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):

        x = self.get_data("intensity", index)
        if self.transform: # Customized transform could be used here for data augmentation
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        y = self.get_data("real", index)
        y = torch.from_numpy(y)

        z = self.get_data("imag", index)
        z = torch.from_numpy(z)
        return (x, y, z)

    def __len__(self):
        return len(self.get_data_infos('intensity'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = -1
                    if load_data:
                        idx = self._add_to_cache(ds.value, file_path)
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = self._add_to_cache(ds[()], file_path)
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)
                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
