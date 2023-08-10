import torch
from datasets import get_dataloader
from caption_model import Detr, params
from configs import build_args_test
import sys
import warnings
warnings.filterwarnings('ignore')

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

    for original, predicted in zip(oText, pText):
        print(f'original: {vocab.decode(original)}')
        print(f'predicted: {vocab.decode(predicted)}')
        print("-"*40)

def get_key_masks(key_, bool_mask=False):
    # key padding mask -> 1 where padding is there 0 otherwise
    mask = torch.ones_like(key_, dtype=torch.float64) # [N, S]
    target_zeros = ~(key_ == 0) # padding values to be set as 1
    mask[target_zeros] = 0

    if bool_mask:
        return mask.bool()

    mask[mask==1] = torch.tensor(float('-inf'))

    return mask

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

    image, text = next(iter(train_data))
    print(image.shape)
    print(text.shape)

    image = image.to(device)
    text = text.to(device)

    predicted_text = model(image, eval_mode=True)

    decode_text(vocab, text, predicted_text)



