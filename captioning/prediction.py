import torch
import wandb
from tqdm import tqdm
from transformers import BertTokenizer

from captioning.utils.utils import NestedTensor


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()

    total = len(data_loader)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    predicted_captions = []
    with tqdm(total=total) as pbar:
        for imgs, masks, captions, cap_masks in data_loader:

            samples = NestedTensor(imgs, masks).to(device)
            captions = captions.to(device)
            cap_masks = cap_masks.to(device)

            predictions = model(samples, captions[:, :-1], cap_masks[:, :-1])
            # predictions = predictions[:, 1, :]
            for k in range(imgs.shape[0]):
                predicted_id = torch.argmax(predictions[k], axis=-1)
                #
                # if predicted_id[0] == 102:
                #     return captions
                #
                # captions[:, i+1] = predicted_id[0]
                # cap_masks[:, i+1] = False

                predicted = tokenizer.decode(
                    predicted_id.tolist(), skip_special_tokens=True
                )
                real_caption = tokenizer.decode(
                    captions[k].tolist(), skip_special_tokens=True
                )
                predicted_captions.append([predicted, real_caption])
                k += 1

        i = 0
        references_corpus = []
        candidate_corpus = []
        for i, e in enumerate(predicted_captions):
            # if i % 5 == 0:
            if i < 10:
                # print('Image name: {}'.format(test_dataloader.dataset.dataset.dataset.names[i]))
                print("Real caption: ", predicted_captions[i][1].split(" "))
                print("Predicted caption: ", predicted_captions[i][0].split(" "))
                print("\n")
            references_corpus.append(predicted_captions[i][1].split(" "))
            candidate_corpus.append(predicted_captions[i][0].split(" "))

        # else:
        #     references_corpus[i // 5].append(predicted_captions[i][1].split(" "))
        # i += 1

        from torchtext.data.metrics import bleu_score

        from captioning.metrics.cider.cider import Cider
        from captioning.metrics.metrics import meteor_score
        from captioning.metrics.Rouge import Rouge

        cider = Cider()
        rouge = Rouge()
        cider_score = cider.compute_score(references_corpus, candidate_corpus)
        print(cider_score)
        rouge_score = rouge.compute_score(references_corpus, candidate_corpus)
        print(rouge_score)
        bleu_score = bleu_score(candidate_corpus, [[e] for e in references_corpus])
        print(bleu_score)
        meteor_score = meteor_score(references_corpus, candidate_corpus)
        print(meteor_score)

        wandb.log(
            {
                "cider_score": cider_score,
                "rouge_score": rouge_score,
                "bleu_score": bleu_score,
                "meteor_score": meteor_score,
            }
        )
