import csv
import itertools
import os
import re
from csv import reader

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import spacy
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from configuration import transformers_config

trf_config = transformers_config()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(report_address, instance_uid_adress, image_path):
    report_file = pd.read_csv(report_address, usecols=["Report"], encoding="utf-8")
    instance_uid = pd.read_csv(instance_uid_adress, usecols=["SOPInstanceUID"])

    # report_file= pd.read_csv("/media/amir_shirian/Amir/Mona/Segmentation/Pneumothorax_reports.csv", usecols=['Report'], encoding='utf-8')
    # instance_uid = pd.read_csv("/media/amir_shirian/Amir/Mona/Segmentation/Pneumothorax_reports.csv", usecols=['SOPInstanceUID'])
    # print(instanceuid)
    # print(len(report))
    # print(len(instanceuid))

    lines = []
    reports = []
    names = []

    for index, row in enumerate(zip(report_file.iterrows(), instance_uid.iterrows())):
        lines = row[0][1].values[0].split("\n")
        name = row[1][1].values[0]

        # check if the name file exists
        if not os.path.exists(os.path.join(image_path, name)):
            continue

        a, b = -1, -1
        for i, rep_line in enumerate(lines):
            if (
                (
                    "chest" in rep_line.lower()
                    or "findings" in rep_line.lower()
                    or "report" in rep_line.lower()
                )
                and "clinical data" not in rep_line.lower()
                and "reported" not in rep_line.lower()
                and a == -1
            ):
                a = i
            if (
                (
                    "transcribed" in rep_line.lower()
                    or "dr." in rep_line.lower()
                    or "unspecified" in rep_line.lower()
                )
                and b == -1
                and a != i
            ):
                b = i
            if a != -1 and b != -1:
                if "chest:" in rep_line.lower():
                    reports.append(
                        "".join(lines[a:b])
                        .replace("  ", " ")
                        .replace(".", ". ")
                        .replace("  ", " ")
                        .strip()
                    )
                    names.append(name)
                else:
                    reports.append(
                        "".join(lines[a + 1 : b])
                        .replace("  ", " ")
                        .replace(".", ". ")
                        .replace("  ", " ")
                        .strip()
                    )
                    names.append(name)
                break

        if a != -1 and b == -1:
            reports.append(
                "".join(lines[a + 1 :])
                .replace("  ", " ")
                .replace(".", ". ")
                .replace("  ", " ")
                .strip()
            )
            names.append(name)
        if reports[-1] == "":
            # print(row['Report'], index)
            # c+=1
            reports.pop(-1)
            names.pop(-1)
    # print(len(reports))
    # print(len(names))
    return reports, names


# df = pd.DataFrame.from_dict({'Name':names, 'Report': reports})
# df.to_csv('data.csv', index=False, header=False)


# Download with: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")


# We want to convert text -> numerical values
# We need a Vocabulary mapping each word to a index
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

    def denumericalize(self, numerical_list):
        return " ".join(
            [self.itos[idx] for idx in numerical_list if idx not in [0, 1, 2, 3]]
        )


# DataSet


class Xray_Report_Dataset(Dataset):
    """Xray image with corresponding report dataset."""

    def __init__(self, names, reports, image_path, freq_threshold, transform=None):
        """
        Args:
            names (list): List of names
            report (list): List of reports
            image_path (string): Directory with all the images.
            freq_threshold (int): Minimum frequency of words to be included in vocabulary.
            transform (callable, optional): Optional transform to be applied
        """
        self.names = names
        self.reports = reports
        # self.reports = pad_sequence(self.reports, batch_first=False, padding_value=0)
        self.image_path = image_path
        self.transform = transform
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.reports)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_path, self.names[idx])
        image = dicom.dcmread(img_name).pixel_array.astype("float")
        image = (np.maximum(image, 0) / image.max()) * 255.0
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # channel last in image

        report = self.reports[idx]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(report)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # image = torch.from_numpy(image)

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "report": torch.tensor(numericalized_caption)}

        return sample


def pad_to_max(reports, max_len):

    # pad first seq to desired length
    reports[0] = nn.ConstantPad1d((0, max_len - reports[0].shape[0]), 0)(reports[0])

    # pad all seqs to desired length
    reports = pad_sequence(reports)

    return reports


# collate_fn that pads the sequences to the max length
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item["image"].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        reports = [item["report"] for item in batch]
        # reports = pad_sequence(reports, batch_first=False, padding_value=self.pad_idx)
        reports = pad_to_max(reports, trf_config.max_length_caption)

        return images, reports


def get_data_loader(
    image_path,
    report_address,
    instance_uid_adress,
    batch_size,
    num_workers=3,
    transform=None,
):
    reports, names = load_data(report_address, instance_uid_adress, image_path)
    dataset = Xray_Report_Dataset(
        names, reports, image_path, freq_threshold=5, transform=transform
    )

    # DataLoader
    train_size = int(len(dataset) * 0.8)
    val_test_size = len(dataset) - train_size
    val_size = int(val_test_size * 0.5)
    test_size = val_test_size - val_size

    pad_idx = dataset.vocab.stoi["<PAD>"]
    training_data, val_test_data = torch.utils.data.random_split(
        dataset,
        [train_size, val_test_size],
        generator=torch.Generator().manual_seed(11),
    )
    validation_data, test_data = torch.utils.data.random_split(
        val_test_data,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(11),
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=num_workers,
    )

    # image, report = next(iter(train_dataloader))
    # # print(report)
    # text_report = train_dataloader.dataset.dataset.vocab.denumericalize(report[:, 0].detach().cpu().numpy())
    # plt.imshow(image[0].detach().cpu().numpy())
    # plt.title(text_report)
    # plt.show()
    # print(image.shape)
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    training_dataloader, validation_dataloader, test_dataloader = get_data_loader(
        report_address="/media/amir_shirian/Amir/Mona/Segmentation/Pneumothorax_reports.csv",
        instance_uid_adress="/media/amir_shirian/Amir/Mona/Segmentation/Pneumothorax_reports.csv",
        batch_size=32,
    )
    print((len(training_dataloader), len(validation_dataloader), len(test_dataloader)))
