import torch
from datasets import get_dataloader
from caption_model import Detr, params
from configs import build_args_test_sample
import sys
import pickle
from PIL import Image
import torchvision
import torchvision.transforms as transforms
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

def decode_text(vocab, pText):
    pText = pText.detach().cpu().numpy()
    for predicted in pText:
        print(f'predicted: {vocab.decode(predicted)}')

def load_img(args):
    img_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor()
    ])
    img = Image.open(args.image_path).convert("RGB")
    img.save('sample_img.png')
    img = img_transform(img)
    img = img.unsqueeze(0)

    return img

if __name__ == "__main__":

    args = build_args_test_sample(sys.argv)
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    args.vocabulary_size = len(vocab)
    model = load_model(args)
    device = torch.device(f'cuda:{args.gpu}')

    model_weights = torch.load(args.model_path, map_location=device)
    model = model.to(device)
    print(model.load_state_dict(model_weights))
    model.eval()

    image = load_img(args)
    image = image.to(device)

    with torch.no_grad():
        predicted_text = model(image, eval_mode=True)
    
    decode_text(vocab, predicted_text)

    



