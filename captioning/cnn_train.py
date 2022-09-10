######### Implement CNN LSTM model for image captioning

import os

import cv2
import pydicom as dicom
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchtext.data import bleu_score

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm


class EncoderCNN(nn.Module):
    def __init__(self, train_CNN, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.encoded_image_size = encoded_image_size

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        # Fine tune
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Only train last 2 layers of resnet if at all required
        if train_CNN:
            for c in list(self.resnet.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = train_CNN

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(
            features
        )  # batch, 512, encoded_image_size, encoded_image_size
        features = features.permute(
            0, 2, 3, 1
        )  # batch, encoded_image_size, encoded_image_size, 512
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(
        self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout
    ):
        """
        decoder_dim is hidden_size for lstm cell
        """
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions):
        """
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        decode_length = encoded_captions.size(1) - 1

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, decode_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)

        for t in range(decode_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(
                self.f_beta(h)
            )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1),
                (h, c),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas


class CNNtoRNN(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=512,
        dropout=0.5,
        train_CNN=False,
    ):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(train_CNN=train_CNN)
        self.decoderRNN = DecoderRNN(
            attention_dim,
            embed_dim,
            decoder_dim,
            vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )

    def forward(self, images, captions):
        encoder_out = self.encoderCNN(images)
        outputs, alphas = self.decoderRNN(encoder_out, captions)
        return outputs, alphas

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = [1]

        with torch.no_grad():
            encoder_out = self.encoderCNN(image)

            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            vocab_size = self.decoderRNN.vocab_size

            encoder_out = encoder_out.view(
                batch_size, -1, encoder_dim
            )  # (batch_size, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # initially start with sos as a predicted word
            predicted = torch.tensor([vocabulary.stoi["<SOS>"]]).to(device)
            h, c = self.decoderRNN.init_hidden_state(
                encoder_out
            )  # (batch_size, decoder_dim)

            for t in range(max_length):
                embeddings = self.decoderRNN.embedding(predicted)  # (1, embed_dim)

                attention_weighted_encoding, alpha = self.decoderRNN.attention(
                    encoder_out, h
                )
                gate = self.decoderRNN.sigmoid(
                    self.decoderRNN.f_beta(h)
                )  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

                h, c = self.decoderRNN.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
                )  # (batch_size_t, decoder_dim)
                preds = self.decoderRNN.fc(
                    self.decoderRNN.dropout(h)
                )  # (batch_size_t, vocab_size)

                predicted = preds.argmax(1)
                result_caption.append(predicted.item())

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.itos[idx] for idx in result_caption]


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


base_path = "/media/amir_shirian/abd1fa2e-8fdf-46e4-9dbf-dc0a070ba9b6/home/user/Desktop/Segmentation/dataset"


def showAndCaptionImage(img, model, transform, vocab):
    img_name = os.path.join(base_path, img)
    img = dicom.dcmread(img_name).pixel_array.astype("float")
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
    img = transform(img)
    # no grad
    with torch.no_grad():
        caption = model.caption_image(img.unsqueeze(0).to(device), vocab)[1:-1]
    captionStr = ""
    for e in caption:
        captionStr += e + " "
    print(captionStr)


def count_parameters(model):
    Num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if Num_param > 1e6:
        Num_param = Num_param / 1e6
        print("Number of parameters: %.6fM" % Num_param)
    else:
        print("Number of parameters: %.6fK" % (Num_param / 1e3))


def train():

    RESIZE = 356
    CROP = 299
    # Train the model
    batch_size = 12
    # number_workers = 4
    number_workers = 0
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((RESIZE, RESIZE)),
            transforms.RandomCrop((CROP, CROP)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load dataloader
    from captioning.data.cnn_dataloader import get_data_loader

    training_dataloader, validation_dataloader, test_dataloader = get_data_loader(
        image_path="//media/amir_shirian/abd1fa2e-8fdf-46e4-9dbf-dc0a070ba9b6/home/user/Desktop/Segmentation/dataset",
        report_address="/media/amir_shirian/abd1fa2e-8fdf-46e4-9dbf-dc0a070ba9b6/home/user/Desktop/Segmentation/Pneumothorax_reports.csv",
        instance_uid_adress="/media/amir_shirian/abd1fa2e-8fdf-46e4-9dbf-dc0a070ba9b6/home/user/Desktop/Segmentation/Pneumothorax_reports.csv",
        batch_size=batch_size,
        transform=transform,
        num_workers=number_workers,
    )

    # Model Hyperparameters
    attention_dim = 700
    embed_dim = 700
    decoder_dim = 700
    dropout = 0.5
    vocab_size = len(training_dataloader.dataset.dataset.vocab)
    learning_rate = 1e-03
    num_epochs = 100
    # num_epochs = 1
    load_model = False
    save_model = True
    train_CNN = True
    alpha_c = 1
    # for tensorboard
    # writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        train_CNN=train_CNN,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=training_dataloader.dataset.dataset.vocab.stoi["<PAD>"]
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        model.load_state_dict(torch.load("my_checkpoint.pth.tar")["state_dict"])
        optimizer.load_state_dict(torch.load("my_checkpoint.pth.tar")["optimizer"])
        step = torch.load("my_checkpoint.pth.tar")["step"]
        # step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        # step = load_checkpoint(torch.load("../input/flickr8k/my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    min_valid_loss = np.inf
    n_epochs_stop = 6
    epochs_no_improve = 0
    early_stop = False
    # for epoch in range(num_epochs):
    #
    #     train_loss = 400
    #
    #     for idx, (imgs, captions) in tqdm(
    #             enumerate(training_dataloader), total=len(training_dataloader), leave=False
    #     ):
    #         imgs = imgs.to(device)
    #         captions = captions.to(device)
    #
    #         outputs, alphas = model(imgs, captions.permute(1, 0))
    #         train_loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.permute(1, 0)[:, 1:].reshape(-1))
    #
    #         # writer.add_scalar("Training loss", loss.item(), global_step=step)
    #         step += 1
    #
    #         optimizer.zero_grad()
    #         train_loss.backward(train_loss)
    #
    #         # Add doubly stochastic attention regularization
    #         # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    #
    #         optimizer.step()
    #
    #
    #
    #
    #     valid_loss = 100
    #     model.eval()
    #     for idx, (imgs, captions) in tqdm(
    #             enumerate(validation_dataloader), total=len(validation_dataloader), leave=False
    #     ):
    #         imgs = imgs.to(device)
    #         captions = captions.to(device)
    #         # no grad
    #         with torch.no_grad():
    #             outputs, alphas = model(imgs, captions.permute(1, 0))
    #         valid_loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.permute(1, 0)[:, 1:].reshape(-1))
    #         valid_loss += valid_loss.item()
    #
    #     print('Epoch {} completed with train loss {} and validation loss {}'.format(epoch + 1, train_loss / len(training_dataloader),
    #                                                        valid_loss / len(validation_dataloader)))
    #
    #     if min_valid_loss > valid_loss:
    #         print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model")
    #         epochs_no_improve = 0
    #         min_valid_loss = valid_loss
    #         if save_model:
    #             checkpoint = {
    #                 "state_dict": model.state_dict(),
    #                 "optimizer": optimizer.state_dict(),
    #                 "step": step,
    #             }
    #             save_checkpoint(checkpoint)
    #             torch.save(model.state_dict(), 'puremodel.pth.tar')
    #
    #     else:
    #         epochs_no_improve += 1
    #     if epoch > 5 and epochs_no_improve == n_epochs_stop:
    #         print('Early stopping!' )
    #         early_stop = True
    #         break
    #     else:
    #         continue

    # test
    model.eval()
    predicted_captions = []
    i = 0

    for idx, (imgs, captions) in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader), leave=False
    ):
        for k in range(imgs.shape[0]):
            img = imgs[k].unsqueeze(0)
            real_caption = [
                test_dataloader.dataset.dataset.dataset.vocab.itos[j.item()]
                for j in captions[:, k]
            ]
            # no grad
            with torch.no_grad():
                predicted_captions.append(
                    [
                        model.caption_image(
                            img.to(device),
                            test_dataloader.dataset.dataset.dataset.vocab,
                        ),
                        real_caption,
                    ]
                )
            i += 1

    references_corpus = []
    candidate_corpus = []
    for i in range(len(predicted_captions)):
        # for e in predicted_captions:
        # print('Image name: {}'.format(test_dataloader.dataset.dataset.dataset.names[i]))
        # print('Real caption: ', [e for e in predicted_captions[i][1] if e != '<PAD>'])
        # print('Predicted caption: ', [e for e in predicted_captions[i][0] if e != '<PAD>'])
        # print('\n')
        references_corpus.append([e for e in predicted_captions[i][1] if e != "<PAD>"])
        candidate_corpus.append([e for e in predicted_captions[i][0] if e != "<PAD>"])

    from captioning.metrics.cider.cider import Cider
    from captioning.metrics.metrics import meteor_score
    from captioning.metrics.Rouge import Rouge

    cider = Cider()
    rouge = Rouge()
    print(cider.compute_score(references_corpus, candidate_corpus))
    print(rouge.compute_score(references_corpus, candidate_corpus))
    print(bleu_score(candidate_corpus, [[e] for e in references_corpus]))
    print(meteor_score(references_corpus, candidate_corpus))

    subjective_images = [
        "0.0.07.546939.73.5.8.8.063105419444496.8628720887274.0",
        "0.0.08.203506.65.2.5.0.32924650882.3608701767376.8",
        "0.0.08.927913.09.1.9.2.25760865681.7507713718798.5",
    ]
    for image in subjective_images:
        showAndCaptionImage(
            image,
            model,
            transform=transform,
            vocab=test_dataloader.dataset.dataset.dataset.vocab,
        )


train()
