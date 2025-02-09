
import pathlib
import json
import random
import h5py
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import pathlib
import json
import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import h5py
import json
import random
import pathlib
from collections import Counter






# Balanced DataLoader using WeightedRandomSampler
def create_balanced_dataloader(dataset, batch_size=32, shuffle=True):
    # Compute class weights based on the frequency of each class
    class_counts = dataset.class_counts
    num_samples = len(dataset)
    
    # Calculate class weights
    class_weights = [1.0 / class_counts[label] for label in dataset.labels]
    
    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(class_weights, num_samples, replacement=True)
    
    # Create the DataLoader with the sampler
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)







import numpy as np
import h5py
import pathlib
import json
from collections import Counter
from sklearn.model_selection import train_test_split

class WESADDataset_Ba(Dataset):
    def __init__(self, data_file: str, config,selected_subjects: list[str], mode: str = "train",
                 max_samples_per_class: int = 500, mapping_file: str = "unique_mapping_wesad.json",
                 verbose: bool = False, modality_masking: bool = False, modality_mask_percentage: float = 0.3,
                 label_drop: bool = False, label_drop_percentage: float = 0.2):
        """
        araguments
            selected_subjects (list[str]): List of subject IDs to include.
            mode (str): Mode of operation ('train', 'valid', 'test').
            max_samples_per_class (int): Maximum samples per class.
            mapping_file (str): Path to save/load unique subject mapping.
            verbose (bool): If True, prints dataset information.
            modality_masking (bool): If True, enable modality masking.
            modality_mask_percentage (float): Percentage of test data to mask the selected modality.
            label_drop (bool): If True, enable label dropping.
            label_drop_percentage (float): Percentage of test data to drop labels.
        """
        if not pathlib.Path(data_file).is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = h5py.File(data_file, 'r')
        self.selected_subjects = selected_subjects
        self.mode = mode
        self.max_samples_per_class = max_samples_per_class
        self.feature_idx = [3, 5, 6]  # Features to use
        self.modality_masking = modality_masking
        self.modality_mask_percentage = modality_mask_percentage
        self.label_drop = label_drop
        self.label_drop_percentage = label_drop_percentage
        #self.training_mode = training_mode
        
        self.jitter_ratio = config.augmentation.jitter_ratio
        self.jitter_scale_ratio =config.augmentation.jitter_scale_ratio
        self.max_segments = config.augmentation.max_seg

        # Load subjects from the dataset
        subjects_raw = self.data['subjects'][:]
        subjects = np.char.decode(subjects_raw.astype('S'), encoding='utf-8').ravel()

        if verbose:
            unique_subjects = set(subjects)
            print(f"Unique subjects in dataset: {unique_subjects}")

        # Load or create subject mapping
        if pathlib.Path(mapping_file).exists():
            with open(mapping_file, "r") as f:
                unique_mapping = json.load(f)
        else:
            unique_mapping = {value: idx for idx, value in enumerate(sorted(set(subjects)))}
            with open(mapping_file, 'w') as f:
                json.dump(unique_mapping, f, indent=4)

        numerical_values = [unique_mapping.get(value) for value in subjects if value]

        # Filter indices for selected subjects
        indices = [i for s in selected_subjects for i in np.argwhere(np.array(subjects) == s).flatten()]

        # Limit samples per class
        class_indices = {}
        for idx in indices:
            class_label = numerical_values[idx]
            if class_label not in class_indices:
                class_indices[class_label] = []
            if len(class_indices[class_label]) < self.max_samples_per_class:
                class_indices[class_label].append(idx)

        indices = [idx for idx_list in class_indices.values() for idx in idx_list]

        if verbose:
            print(f"Filtered indices count before splitting: {len(indices)}")

        if len(indices) == 0:
            raise ValueError("No valid samples found after filtering.")

        # Pass 'subjects' explicitly here
        self.random_indices, self.labels = self._split_data(indices, numerical_values, unique_mapping, subjects, verbose)

        # Set the num_samples attribute
        self.num_samples = len(self.random_indices)

        # Compute class counts
        self.class_counts = self.compute_class_counts()

        # Precompute indices for masking and label dropping (only for test)
        if self.mode == "test":
            if self.modality_masking:
                # Randomly select one modality to mask
                self.selected_modality = np.random.randint(0, len(self.feature_idx))
                self.mask_indices = np.random.choice(
                    self.num_samples,
                    int(self.num_samples * self.modality_mask_percentage),
                    replace=False
                )
            else:
                self.mask_indices = []

            if self.label_drop:
                self.drop_indices = np.random.choice(
                    self.num_samples,
                    int(self.num_samples * self.label_drop_percentage),
                    replace=False
                )
            else:
                self.drop_indices = []
        else:
            self.mask_indices = []
            self.drop_indices = []

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> tuple:
        if self.mode == "test":
            if i >= len(self.pairs):
                raise IndexError(f"Index {i} out of range for pairs (len: {len(self.pairs)})")

            idx1, idx2 = self.pairs[i]
            data1 = self.data['features'][idx1].T
            data2 = self.data['features'][idx2].T
            label = self.pair_labels[i]

            # Mask the selected modality for selected test samples
            if self.modality_masking and i in self.mask_indices:
                data1[:, self.selected_modality] = 0  # Mask the selected modality
                data2[:, self.selected_modality] = 0  # Mask the selected modality

            # Drop label for selected test samples
            if self.label_drop and i in self.drop_indices:
                label = -1  # Use -1 or any other placeholder for missing labels

            return data1[:, self.feature_idx], data2[:, self.feature_idx], label
        else:
            random_index = self.random_indices[i]
            data = self.data['features'][random_index].T
            label = self.labels[i]
            data=torch.from_numpy(data)
                    #print(data.size())
            #if self.mode == "train":
               
            data = data + self.jitter_ratio*torch.randn(data.size())
            #data[:, self.feature_idx]
            #print('shape',data.shape)
            #data[self.feature_idx, :]
                        #data[:, self.feature_idx]
            return data[:, self.feature_idx], label

    def compute_class_counts(self) -> dict[int, int]:
        return dict(Counter(self.labels))

    def count_unique_labels(self) -> int:
        return len(set(self.labels))

    def _split_data(self, indices, numerical_values, unique_mapping, subjects, verbose):
        """Split data into train, validation, and test sets."""
        train_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=24)
        train_indices, valid_indices = train_test_split(train_indices, test_size=0.1, random_state=24)

        if self.mode == "train":
            random_indices = train_indices
        elif self.mode == "valid":
            random_indices = valid_indices
        elif self.mode == "test":
            random_indices = test_indices
            self.pairs, self.pair_labels = self._create_test_pairs(test_indices, numerical_values, verbose)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train', 'valid', or 'test'.")

        # Corrected line: passing 'subjects' to _split_data
        labels = [unique_mapping.get(subjects[idx]) for idx in random_indices if subjects[idx]]
        return random_indices, labels

    def _create_test_pairs(self, test_indices, numerical_values, verbose):
        """Create pairs for test mode."""
        pairs, pair_labels = [], []
        similar_count, dissimilar_count = 0, 0

        for i in range(len(test_indices)):
            for j in range(i + 1, len(test_indices)):
                if numerical_values[test_indices[i]] == numerical_values[test_indices[j]]:
                    pairs.append((test_indices[i], test_indices[j]))
                    pair_labels.append(0)
                    similar_count += 1
                else:
                    pairs.append((test_indices[i], test_indices[j]))
                    pair_labels.append(1)
                    dissimilar_count += 1

        if verbose:
            print(f"Total similar pairs: {similar_count}")
            print(f"Total dissimilar pairs: {dissimilar_count}")

        return pairs, pair_labels



import numpy as np
import h5py
import pathlib
import json
from collections import Counter
from sklearn.model_selection import train_test_split

class VERBIODataset_train_valid_Ba(Dataset):
    def __init__(self, data_file: str, selected_subjects: list[str], mode: str = "train",
                 max_samples_per_class: int = 1000, mapping_file: str = "unique_mapping_verbio.json",
                 verbose: bool = False, modality_masking: bool = False, modality_mask_percentage: float = 0.3,
                 label_drop: bool = False, label_drop_percentage: float = 0.2):
        """
        arguments
            data_file (str): Path to the data file containing the dataset.
            selected_subjects (list[str]): List of subject IDs to include.
            mode (str): Mode of operation ('train', 'valid', 'test').
            max_samples_per_class (int): Maximum samples per class.
            mapping_file (str): Path to save/load unique subject mapping.
            verbose (bool): If True, prints dataset information.
            modality_masking (bool): If True, enable modality masking.
            modality_mask_percentage (float): Percentage of test data to mask the selected modality.
            label_drop (bool): If True, enable label dropping.
            label_drop_percentage (float): Percentage of test data to drop labels.
        """
        if not pathlib.Path(data_file).is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = h5py.File(data_file, 'r')
        self.selected_subjects = selected_subjects
        self.mode = mode
        self.max_samples_per_class = max_samples_per_class
        self.feature_idx = [3, 5, 6]  # Features to use
        self.modality_masking = modality_masking
        self.modality_mask_percentage = modality_mask_percentage
        self.label_drop = label_drop
        self.label_drop_percentage = label_drop_percentage

        # Load subjects from the dataset
        subjects_raw = self.data['subjects'][:]
        subjects = np.char.decode(subjects_raw.astype('S'), encoding='utf-8').ravel()

        if verbose:
            unique_subjects = set(subjects)
            print(f"Unique subjects in dataset: {unique_subjects}")

        # Load or create subject mapping
        if pathlib.Path(mapping_file).exists():
            with open(mapping_file, "r") as f:
                unique_mapping = json.load(f)
        else:
            unique_mapping = {value: idx for idx, value in enumerate(sorted(set(subjects)))}
            with open(mapping_file, 'w') as f:
                json.dump(unique_mapping, f, indent=4)

        numerical_values = [unique_mapping.get(value) for value in subjects if value]

        # Filter indices for selected subjects
        indices = [i for s in selected_subjects for i in np.argwhere(np.array(subjects) == s).flatten()]

        # Limit samples per class
        class_indices = {}
        for idx in indices:
            class_label = numerical_values[idx]
            if class_label not in class_indices:
                class_indices[class_label] = []
            if len(class_indices[class_label]) < self.max_samples_per_class:
                class_indices[class_label].append(idx)

        indices = [idx for idx_list in class_indices.values() for idx in idx_list]

        if verbose:
            print(f"Filtered indices count before splitting: {len(indices)}")

        if len(indices) == 0:
            raise ValueError("No valid samples found after filtering.")

        # Pass 'subjects' explicitly here
        self.random_indices, self.labels = self._split_data(indices, numerical_values, unique_mapping, subjects, verbose)

        # Set the num_samples attribute
        self.num_samples = len(self.random_indices)

        # Compute class counts
        self.class_counts = self.compute_class_counts()

        # Precompute indices for masking and label dropping (only for test)
        if self.mode == "test":
            if self.modality_masking:
                # Randomly select one modality to mask
                self.selected_modality = np.random.randint(0, len(self.feature_idx))
                self.mask_indices = np.random.choice(
                    self.num_samples,
                    int(self.num_samples * self.modality_mask_percentage),
                    replace=False
                )
            else:
                self.mask_indices = []

            if self.label_drop:
                self.drop_indices = np.random.choice(
                    self.num_samples,
                    int(self.num_samples * self.label_drop_percentage),
                    replace=False
                )
            else:
                self.drop_indices = []
        else:
            self.mask_indices = []
            self.drop_indices = []

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> tuple:
        if self.mode == "test":
            if i >= len(self.pairs):
                raise IndexError(f"Index {i} out of range for pairs (len: {len(self.pairs)})")

            idx1, idx2 = self.pairs[i]
            data1 = self.data['features'][idx1].T
            data2 = self.data['features'][idx2].T
            label = self.pair_labels[i]

            # Mask the selected modality for selected test samples
            if self.modality_masking and i in self.mask_indices:
                data1[:, self.selected_modality] = 0  # Mask the selected modality
                data2[:, self.selected_modality] = 0  # Mask the selected modality

            # Drop label for selected test samples
            if self.label_drop and i in self.drop_indices:
                label = -1  # Use -1 or any other placeholder for missing labels

            return data1[:, self.feature_idx], data2[:, self.feature_idx], label
        else:
            random_index = self.random_indices[i]
            data = self.data['features'][random_index].T
            label = self.labels[i]
            data=torch.from_numpy(data)
                  
               
            data = data + self.jitter_ratio*torch.randn(data.size())
            return data[:, self.feature_idx], label

    def compute_class_counts(self) -> dict[int, int]:
        return dict(Counter(self.labels))

    def count_unique_labels(self) -> int:
        return len(set(self.labels))

    def _split_data(self, indices, numerical_values, unique_mapping, subjects, verbose):
        """Split data into train, validation, and test sets."""
        train_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=24)
        train_indices, valid_indices = train_test_split(train_indices, test_size=0.1, random_state=24)

        if self.mode == "train":
            random_indices = train_indices
        elif self.mode == "valid":
            random_indices = valid_indices
        elif self.mode == "test":
            random_indices = test_indices
            self.pairs, self.pair_labels = self._create_test_pairs(test_indices, numerical_values, verbose)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train', 'valid', or 'test'.")

        # Corrected line: passing 'subjects' to _split_data
        labels = [unique_mapping.get(subjects[idx]) for idx in random_indices if subjects[idx]]
        return random_indices, labels

    def _create_test_pairs(self, test_indices, numerical_values, verbose):
        """Create pairs for test mode."""
        pairs, pair_labels = [], []
        similar_count, dissimilar_count = 0, 0

        for i in range(len(test_indices)):
            for j in range(i + 1, len(test_indices)):
                if numerical_values[test_indices[i]] == numerical_values[test_indices[j]]:
                    pairs.append((test_indices[i], test_indices[j]))
                    pair_labels.append(0)
                    similar_count += 1
                else:
                    pairs.append((test_indices[i], test_indices[j]))
                    pair_labels.append(1)
                    dissimilar_count += 1

        if verbose:
            print(f"Total similar pairs: {similar_count}")
            print(f"Total dissimilar pairs: {dissimilar_count}")

        return pairs, pair_labels







# Example usage
if __name__ == "__main__":

    selected_subjectsw=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    selected_subjectss=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
    selected_subjects=[
                   'P001', 'P003', 'P004', 'P005', 'P006',
                   'P007', 'P008', 'P009','P011', 'P012',
                   'P013', 'P014', 'P016', 'P017','P018',
                   'P020','P021', 'P023', 'P026', 'P027',
                   'P031','P032','P035','P037','P038',
                   'P039','P040','P041','P042','P043',
                   'P044','P045','P046','P047','P048',
                   'P049','P050','P051','P052','P053',
                   'P056','P057','P058','P060','P061',
                   'P062','P063','P064','P065','P066',
                   'P067','P068','P071','P072','P073'
               ]



   
    path_1B= '/VerBIO-norm.hdf5'
    
    path_1= '/WESAD-norm.hdf5'
    selected_subjects = ['S1','S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
    
    
    # Create datasets
    train_dataset = WESADDataset_Ba(path_1, selected_subjects, mode="train")
    test_dataset = WESADDataset_Ba(path_1, selected_subjects, mode="test",
                               modality_masking=False, modality_mask_percentage=0.3,
                               label_drop=False, label_drop_percentage=0.2)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Iterate through the test DataLoader
    for batch in train_loader:
        data1, labels = batch
        print("Test batch - Data1:", data1.shape, "Labels:", labels)
        break
    
    
    
    
    
    unique_label_count = test_dataset.count_unique_labels()
    print("Number of unique labels:", unique_label_count)
    
    #for (data1,data2, label2) in train_loader:
        # Process the pair data and labels
        #print(label2)
    


