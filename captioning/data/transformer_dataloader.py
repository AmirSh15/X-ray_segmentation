import os
import random

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import spacy
import torch
import torchvision as tv
import torchvision.transforms.functional as TF
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from captioning.utils.utils import nested_tensor_from_tensor_list, read_json
from configuration import transformers_config

trf_config = transformers_config()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DIM = 299


def under_max(image):
    image = dicom.dcmread(image).pixel_array.astype("float")
    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # if image.mode != 'RGB':
    #     image = image.convert("RGB")
    #
    # shape = np.array(image.size, dtype=np.float)
    # long_dim = max(shape)
    # scale = MAX_DIM / long_dim
    #
    # new_shape = (shape * scale).astype(int)
    # image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose(
    [
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((trf_config.RESIZE, trf_config.RESIZE)),
        tv.transforms.RandomCrop((trf_config.CROP, trf_config.CROP)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(),
        tv.transforms.ToTensor(),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


val_transform = tv.transforms.Compose(
    [
        # tv.transforms.Lambda(under_max),
        tv.transforms.ToTensor(),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


class CocoCaption(Dataset):
    def __init__(
        self, root, ann, max_length, limit, transform=train_transform, mode="training"
    ):
        super().__init__()

        self.root = root
        self.transform = transform
        # self.annot = [(self._process(val['image_id']), val['caption'])
        #               for val in ann['annotations']]
        self.annot = ann
        if mode == "validation":
            self.annot = self.annot
        if mode == "training":
            self.annot = self.annot[:limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower=True
        )
        self.max_length = max_length + 1

    # def _process(self, image_id):
    #     val = str(image_id).zfill(12)
    #     return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        # image = Image.open(os.path.join(self.root, image_id))
        image = os.path.join(self.root, image_id)
        image = under_max(image)

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
        )

        caption = np.array(caption_encoded["input_ids"])
        cap_mask = (1 - np.array(caption_encoded["attention_mask"])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode="training"):
    if mode == "training":
        train_dir = os.path.join(config.dir, "train2017")
        train_file = os.path.join(config.dir, "annotations", "captions_train2017.json")
        data = CocoCaption(
            train_dir,
            read_json(train_file),
            max_length=config.max_position_embeddings,
            limit=config.limit,
            transform=train_transform,
            mode="training",
        )
        return data

    elif mode == "validation":
        val_dir = os.path.join(config.dir, "val2017")
        val_file = os.path.join(config.dir, "annotations", "captions_val2017.json")
        data = CocoCaption(
            val_dir,
            read_json(val_file),
            max_length=config.max_position_embeddings,
            limit=config.limit,
            transform=val_transform,
            mode="validation",
        )
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")


def load_data(report_address, instance_uid_adress, image_path):
    report_file = pd.read_csv(report_address, usecols=["Report"], encoding="utf-8")
    instance_uid = pd.read_csv(instance_uid_adress, usecols=["SOPInstanceUID"])

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
    merged_reports_names = [(names[i], reports[i]) for i in range(0, len(reports))]
    return merged_reports_names


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
        # numericalized_caption = [self.vocab.stoi["<SOS>"]]
        # numericalized_caption += self.vocab.numericalize(report)
        # numericalized_caption.append(self.vocab.stoi["<EOS>"])

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower=True)
        max_length = trf_config.max_length_caption + 1
        report_encoded = tokenizer.encode_plus(
            report,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
        )

        numericalized_caption = np.array(report_encoded["input_ids"])
        cap_mask = (1 - np.array(report_encoded["attention_mask"])).astype(bool)

        # image = torch.from_numpy(image)

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        image_nested = nested_tensor_from_tensor_list(image.unsqueeze(0))
        image = image_nested.tensors.squeeze(0)
        img_mask = image_nested.mask.squeeze(0)

        sample = {
            "image": image,
            "mask": img_mask,
            "report": numericalized_caption,
            "cap_mask": cap_mask,
        }

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
        masks = [item["mask"] for item in batch]
        reports = [item["report"] for item in batch]
        caps_mask = [item["cap_mask"] for item in batch]
        # reports = pad_sequence(reports, batch_first=False, padding_value=self.pad_idx)
        # reports = pad_to_max(reports, trf_config.max_length_caption)

        return images, masks, reports, caps_mask


def get_data_loader(
    image_path,
    report_address,
    instance_uid_adress,
    batch_size,
    num_workers=3,
    transform=None,
):
    data = load_data(report_address, instance_uid_adress, image_path)
    dataset = CocoCaption(
        image_path,
        data,
        max_length=trf_config.max_length_caption,
        limit=trf_config.limit,
        transform=train_transform,
        mode="training",
    )
    # dataset = Xray_Report_Dataset(data, image_path, freq_threshold=5, transform=transform)
    # dataset = Xray_Report_Dataset(names, reports, image_path, freq_threshold=5, transform=transform)

    # DataLoader
    train_size = int(len(dataset) * 0.8)
    val_test_size = len(dataset) - train_size
    val_size = int(val_test_size * 0.5)
    test_size = val_test_size - val_size
    # pad_idx = dataset.vocab.stoi["<PAD>"]

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

    sampler_train = torch.utils.data.RandomSampler(training_data)
    sampler_val = torch.utils.data.SequentialSampler(validation_data)
    sampler_test = torch.utils.data.SequentialSampler(test_data)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, trf_config.batch_size, drop_last=True
    )

    train_dataloader = DataLoader(
        training_data,
        batch_sampler=batch_sampler_train,
        num_workers=trf_config.num_workers,
    )

    valid_dataloader = DataLoader(
        validation_data,
        trf_config.batch_size,
        sampler=sampler_val,
        drop_last=False,
        num_workers=trf_config.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        trf_config.batch_size,
        sampler=sampler_test,
        drop_last=False,
        num_workers=trf_config.num_workers,
    )

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    training_dataloader, validation_dataloader, test_dataloader = get_data_loader(
        report_address=trf_config.args.data_path+"/Pneumothorax_reports.csv",
        instance_uid_adress=trf_config.args.data_path+"/Pneumothorax_reports.csv",
        batch_size=32,
    )
    print((len(training_dataloader), len(validation_dataloader), len(test_dataloader)))
