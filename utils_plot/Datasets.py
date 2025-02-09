import h5py
import numpy as np
import random
import json
import pathlib
import itertools
from torch.utils.data import Dataset

class SWEETDataset_pair(Dataset):
    def __init__(self, hdf5_file, selected_subjects, mode="train", max_genuine_per_class=1000, max_impostor_per_class=1000):
        if not pathlib.Path(hdf5_file).is_file():
            raise FileNotFoundError(f'HDF5 file not found: {hdf5_file}')

        self.data = h5py.File(hdf5_file, 'r')
        self.feature_names = self.data['features'].attrs['feature_names'].split(' ')
        self.selected_subjects = selected_subjects
        self.mode = mode
        self.max_genuine_per_class = max_genuine_per_class
        self.max_impostor_per_class = max_impostor_per_class

        subjects = np.array([i.decode() for i in np.array(self.data['subjects'][:]).squeeze()])
        print('Subjects:', subjects)

        unique_mapping = {value: idx for idx, value in enumerate(sorted(set(subjects)))}
        with open('unique_mapping.json', 'w') as f:
            json.dump(unique_mapping, f, indent=4)

        numerical_values = [unique_mapping[value] for value in subjects]
        print('Selected subjects:', selected_subjects)

        try:
            with open("unique_mapping_1.json", "r") as f:
                self.loaded_mapping = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("unique_mapping_1.json not found")

        subjects = np.array(numerical_values)
        indices = [i for s in selected_subjects for i in np.argwhere(subjects == s).flatten()]
        indices = sorted(indices)

        random.seed(24)
        random.shuffle(indices)

        train_test_split = int(0.3 * len(indices))
        train_test_indices = torch.randperm(len(indices)).tolist()
        train_indices = train_test_indices[:train_test_split]
        test_indices = train_test_indices[train_test_split:]

        train_size = int(0.8 * train_test_split)
        train_valid_indices = torch.randperm(len(train_indices)).tolist()

        if self.mode == 'train':
            self.indices = sorted(train_valid_indices[:train_size])
        elif self.mode == 'valid':
            self.indices = sorted(train_valid_indices[train_size:])
        elif self.mode == 'test':
            self.indices = sorted(test_indices)
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'valid', or 'test'.")

        self.num_samples = len(self.indices)
        all_labels = np.hstack(self.data['subjects'][self.indices])
        self.labels = [self.loaded_mapping.get(subject.decode(), None) for subject in all_labels]
        self.labels = [label for label in self.labels if label is not None]

        if self.mode == "test":
            self.pairs, self.pair_labels = self.generate_pairs(self.indices, self.labels, max_total_pairs=self.num_samples)

        print(f'SWEET Dataset: selected_subjects, n={len(self.selected_subjects)}, #instances={self.num_samples}')

    def generate_pairs(self, indices, labels, max_total_pairs=None):
        label_to_indices = {}
        for idx, label in zip(indices, labels):
            label_to_indices.setdefault(label, []).append(idx)

        genuine_pairs = []
        for idx_list in label_to_indices.values():
            if len(idx_list) > 1:
                pairs = list(itertools.combinations(idx_list, 2))
                genuine_pairs.extend(random.sample(pairs, min(self.max_genuine_per_class, len(pairs))))

        impostor_pairs = []
        label_list = list(label_to_indices.keys())
        for i, j in itertools.combinations(range(len(label_list)), 2):
            label1, label2 = label_list[i], label_list[j]
            pairs = list(itertools.product(label_to_indices[label1], label_to_indices[label2]))
            impostor_pairs.extend(random.sample(pairs, min(self.max_impostor_per_class, len(pairs))))

        # Limit total number of pairs if specified
        if max_total_pairs:
            total_pairs = genuine_pairs + impostor_pairs
            if len(total_pairs) > max_total_pairs:
                total_pairs = random.sample(total_pairs, max_total_pairs)
                genuine_pairs = [pair for pair in total_pairs if pair in genuine_pairs]
                impostor_pairs = [pair for pair in total_pairs if pair in impostor_pairs]

        genuine_labels = [1] * len(genuine_pairs)
        impostor_labels = [0] * len(impostor_pairs)

        total_pairs = genuine_pairs + impostor_pairs
        total_labels = genuine_labels + impostor_labels

        # Shuffle pairs and labels together
        combined = list(zip(total_pairs, total_labels))
        random.shuffle(combined)
        total_pairs, total_labels = zip(*combined)

        print(f"Generated {len(genuine_pairs)} genuine and {len(impostor_pairs)} impostor pairs.")

        # Save pairs, labels, and class information to JSON
        index_to_class = {idx: label for idx, label in zip(indices, labels)}
        pairs_data = {
            "pairs": [
                {
                    "indices": list(pair),
                    "classes": [index_to_class[pair[0]], index_to_class[pair[1]]],
                    "label": label
                } for pair, label in zip(total_pairs, total_labels)
            ]
        }
        with open('generated_pairs.json', 'w') as f:
            json.dump(pairs_data, f, indent=4)

        return list(total_pairs), list(total_labels)

    def __len__(self):
        return len(self.pairs) if self.mode == "test" else self.num_samples

    def __getitem__(self, i):
        if self.mode == "test":
            (idx1, idx2), pair_label = self.pairs[i], self.pair_labels[i]
            data1 = self.data['features'][idx1, :, :].T
            data2 = self.data['features'][idx2, :, :].T
            return (data1, data2), pair_label
        else:
            k = self.indices[i]
            data = self.data['features'][k, :, :].T
            return data
