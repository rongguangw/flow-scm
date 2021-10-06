import os
import re
import glob
import time
import torch
import numpy as np
import pandas as pd
import scipy.misc as m
from torch.utils import data
from math import isnan, isinf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.distributions.utils import probs_to_logits


class featuresDataset(data.Dataset):

    def __init__(
        self,
        filename,
        feature_dim = 145,
        inference = False,
        train = True,
        train_ratio = 0.8,
        random_seed = 42,
        zscore = True
    ):
        super(featuresDataset, self).__init__()
        self.filename = filename
        self.data_path = os.path.join('./datasets', self.filename)
        self.data = pd.read_csv(self.data_path, low_memory=False)
        self.feature_dim = feature_dim
        self.random_seed = random_seed
        self.inference = inference
        self.zscore = zscore
        self.label_idx()
        self.flow_prepare()

        self.image_list, self.y = [], []
        print('==> start preprocessing csv ...')
        for index, row in self.data.iterrows():
            self.image_list.append(dict(
                image = np.asarray(row[-self.feature_dim:]).astype(np.float32),
                image_id = row['PTID'],
                sex = self.sex_dict.transform([row['Sex']]),
                age = round(row['Age'], 2),
                scanner = self.scanner_dict.transform([row['Site']])
            ))
        print('==> finished preprocessing csv ...')

        if not self.inference:
            self.img_idxes = np.arange(0, len(self.image_list))
            np.random.seed(self.random_seed)
            np.random.shuffle(self.img_idxes)
            last_train_sample = int(len(self.img_idxes) * train_ratio)
            if train:
                self.img_idxes = self.img_idxes[:last_train_sample]
            else:
                self.img_idxes = self.img_idxes[last_train_sample:]
        else:
            self.img_idxes = np.arange(0, len(self.image_list))

    def label_idx(self):
        self.sex_dict = LabelEncoder()
        self.sex_dict.fit(self.data.Sex.tolist())
        self.sex_class = len(self.sex_dict.classes_)
        self.scanner_dict = LabelEncoder()
        self.scanner_dict.fit(self.data.Site.tolist())
        self.scanner_class = len(self.scanner_dict.classes_)
        self.covariates_dict = {'sex': self.sex_dict,
                                'scanner': self.scanner_dict}

    def flow_prepare(self):
        # sex
        sex_dict = LabelEncoder()
        sex_dict.fit(self.data.Sex.tolist())
        sex = sex_dict.transform(self.data.Sex.tolist())
        sex_counts = Counter(sex)
        sex_mass = [v/sum(sex_counts.values()) for k,v in sex_counts.items()]
        sex_logits = probs_to_logits(torch.as_tensor(sex_mass), is_binary=False)
        # age
        age_mean = torch.as_tensor(self.data.Age.to_numpy()).log().mean()
        age_std = torch.as_tensor(self.data.Age.to_numpy()).log().std()
        # scanner
        scanner = self.scanner_dict.transform(self.data.Site.tolist())
        scanner_counts = Counter(scanner)
        scanner_mass = [v/sum(scanner_counts.values()) for k,v in scanner_counts.items()]
        scanner_logits = probs_to_logits(torch.as_tensor(scanner_mass), is_binary=False)
        # roi
        mean_list, std_list = [], []
        n = len(self.data.columns) - self.feature_dim
        for i in range(self.feature_dim):
            mean = torch.as_tensor(self.data[self.data.columns[n+i]].to_numpy()).log().mean()
            mean_list.append(mean)
            std = torch.as_tensor(self.data[self.data.columns[n+i]].to_numpy()).log().std()
            std_list.append(std)
        filtered_mean = [x for x in mean_list if not isnan(x) and not isinf(x)]
        filtered_std = [x for x in std_list if not isnan(x) and not isinf(x)]
        roi_mean = torch.as_tensor(filtered_mean).mean()
        roi_std = torch.as_tensor(filtered_std).mean()

        self.flow_dict = {'sex_logits': sex_logits.unsqueeze(0).float(),
                          'age_mean': age_mean.float(),
                          'age_std': age_std.float(),
                          'scanner_logits': scanner_logits.unsqueeze(0).float(),
                          'roi_mean': roi_mean.float(),
                          'roi_std': roi_std.float()}

    def zScoreNorm(self, img, min_max=True):
        if min_max:
            res = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            res = (img - np.min(img)) / np.max(img)
            res = (res * 1.0 - np.mean(res)) / np.std(res)
        return res

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):
        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]
        img = img_info['image']

        if self.zscore:
            img = self.zScoreNorm(img)

        id = img_info['image_id']
        img = torch.from_numpy(img).float()
        sex = torch.tensor(img_info['sex']).float()
        age = torch.tensor(img_info['age']).unsqueeze(-1).float()
        scanner = torch.tensor(img_info['scanner']).float()

        if self.inference:
            return img, age, sex, scanner, id
        else:
            return img, age, sex, scanner


def get_features_dataset(
    filename,
    feature_dim=145,
    inference=False,
    train_ratio=0.8,
    random_seed=42,
    zscore=True
    ):

    if not inference:
        dataset_train = featuresDataset(train=True, filename=filename, feature_dim=feature_dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)
        dataset_test = featuresDataset(train=False, filename=filename, feature_dim=feature_dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)

        return dataset_train, dataset_test
    else:
        dataset = featuresDataset(inference=True, filename=filename, feature_dim=feature_dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)

        return dataset
