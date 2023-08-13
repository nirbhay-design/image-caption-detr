import torch
from datasets import get_dataloader
from caption_model import Detr, params
from configs import build_args_test
import sys
from torchmetrics.text import BLEUScore
import warnings
warnings.filterwarnings('ignore')

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f"\r|{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='')
    if (current == total):
        print()

def load_model(args):
    detr = Detr(
        backbone_layers=args.backbone_layers,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        vocab_size=args.vocabulary_size,
    )

    print(f'# of parameters: {params(detr)}')

    return detr

def decode_text(vocab, oText, pText):
    oText = oText.detach().cpu().numpy()
    pText = pText.detach().cpu().numpy()

    original_batch = []
    predicted_batch = []

    for original, predicted in zip(oText, pText):
        # print(f'original: {vocab.decode(original)}')
        # print(f'predicted: {vocab.decode(predicted)}')
        # print("-"*40)
        original_batch.append(vocab.decode(original))
        predicted_batch.append(vocab.decode(predicted))
    
    return original_batch, predicted_batch

@torch.no_grad()
def calculate_BLEU(model, data, device, return_logs=False):
    model = model.to(device)

    outputs = {
        'ground-truths': [],
        'predicted':[]
    }

    len_data = len(data)

    bleu = BLEUScore(n_gram=1)

    for idx, (image, text) in enumerate(data):
        image,text = image.to(device), text.to(device)

        pred_text = model(image, eval_mode=True)

        original_text, predicted_text = decode_text(vocab, text, pred_text)
        original_text = [[i] for i in original_text]

        outputs['ground-truths'].extend(original_text)
        outputs['predicted'].extend(predicted_text)

        if return_logs:
            progress(idx+1, len_data)
    
    return bleu(outputs['predicted'], outputs['ground-truths'])


if __name__ == "__main__":

    args = build_args_test(sys.argv)
    train_data, vocab = get_dataloader(args)
    args.vocabulary_size = len(vocab)
    model = load_model(args)
    device = torch.device(f'cuda:{args.gpu}')

    model_weights = torch.load(args.model_path, map_location=device)
    model = model.to(device)
    print(model.load_state_dict(model_weights))
    model.eval()

    bleu_score = calculate_BLEU(model, train_data, device, return_logs=args.return_logs)

    print(bleu_score) # 44.73, 50.32
    # image, text = next(iter(train_data))
    # print(image.shape)
    # print(text.shape)

    # image = image.to(device)
    # text = text.to(device)

    # with torch.no_grad():
    #     predicted_text = model(image, eval_mode=True)

    # decode_text(vocab, text, predicted_text)



