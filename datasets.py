import torch
import torchvision
import torchtext
import re, string
import os
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from torchtext.data import get_tokenizer

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f"\r|{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='')
    if (current == total):
        print()

class Vocabulary():
    def __init__(self, feq, tok):
        self.feq = feq
        self.itos = {
            0:"<PAD>",
            1:"<SOS>",
            2:"<EOS>",
            3:"<UNK>"
        }
        self.stoi = {j:i for i,j in self.itos.items()}
        self.tok = tok

    def tokenizer(self, text):
        return [tok.text for tok in self.tok.tokenizer(text)]

    def build_voc(self, text_list):
        idx = 4
        curfeq = {}
        for text in text_list:
            for word in self.tokenizer(text):
                if word not in curfeq:
                    curfeq[word] = 1
                else:
                    curfeq[word] += 1
                if curfeq[word] == self.feq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numeric(self, text):
        tokenize_text = self.tokenizer(text)

        numeric_val = [self.stoi['<SOS>']]
        numeric_val += [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenize_text]
        numeric_val += [self.stoi['<EOS>']]

        return numeric_val

    def __len__(self):
        return len(self.itos)

    def decode(self, numeric_val):
        return ' '.join([self.itos[i] for i in numeric_val])

class Flickr30k_data():
    def __init__(self, 
                image_path: str,
                caption_path: str,
                vocab_eng: Vocabulary,
                image_transforms: transforms
                ):
        
        self.image_path = image_path
        self.caption_path = caption_path # caption may have different text for same image
        self.vocab = vocab_eng
        self.img_transform = image_transforms

        jpg_images = list(filter(
            lambda x: '.jpg' in x,
            os.listdir(self.image_path)
        ))

        img_caption_pair_dict: dict[str, list[str]] = {}

        with open(self.caption_path, 'r') as f:
            first_redundant_line = f.readline()
            img_caption_pairs = f.readlines()

            for img_caption_pair in img_caption_pairs:
                image_name, caption = self._extract_img_caption(img_caption_pair)
                if caption is None:
                    continue
                img_caption_pair_dict[image_name] = [*img_caption_pair_dict.get(image_name,[]), caption]
        
        self.data = img_caption_pair_dict.items()

    def preprocess(self, text):
        text = text.lower()#converting string to lowercase
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res
    
    def _extract_img_caption(self, text):
        text = text.replace("\n", '')
        re_txt = r'(\d+.jpg),(.+)?'
        img_caption_match = re.search(re_txt, text)
        img_name = img_caption_match.group(1)
        caption_ = img_caption_match.group(2)
        if caption_ is None:
            return img_name, caption_
        processed_caption = self.preprocess(caption_)
        return img_name, processed_caption 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, captions_list = self.data[idx]
        
        return image_name, captions_list

# class CustomCollate():
#     def __init__(self, pad_idx):
#         self.pad_idx = pad_idx

#     def __call__(self, batch):

#         text_ger = []
#         text_eng = []

#         for bt in batch:
#             text_ger.append(bt[0])
#             text_eng.append(bt[1])

#         padded_text_ger = torch.nn.utils.rnn.pad_sequence(text_ger, batch_first = True, padding_value = self.pad_idx)
#         padded_text_eng = torch.nn.utils.rnn.pad_sequence(text_eng, batch_first = True, padding_value = self.pad_idx)

#         return padded_text_ger, padded_text_eng

# def dataloaders():
#     mt30k_train = multi30k_data(typ='train')
#     eng_voc = mt30k_train.vocab_eng
#     ger_voc = mt30k_train.vocab_ger

#     mt30k_test = multi30k_data(
#         typ='valid',
#         vocab_eng=eng_voc,
#         vocab_ger=ger_voc
#     )

#     pad_idx = 0

#     train_loader = DataLoader(
#         mt30k_train,
#         batch_size=32,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=4,
#         collate_fn = CustomCollate(pad_idx)
#     )

#     test_loader = DataLoader(
#         mt30k_test,
#         batch_size=32,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=4,
#         collate_fn = CustomCollate(pad_idx)
#     )

#     return train_loader, test_loader, ger_voc, eng_voc

if __name__ == "__main__":
    tokenizer = get_tokenizer('basic_english')
    img_transforms = transforms.Compose([
        transforms.Resize((256, int(256 * 1.33))), # (h,w)
        transforms.ToTensor()
    ])
    vocab = Vocabulary(2,tokenizer)

    image_path = "/DATA/dataset/Flickr30k/Flickr30k/Images"
    captions_path = "/DATA/dataset/Flickr30k/Flickr30k/captions.txt"


    data = Flickr30k_data(
        image_path,
        captions_path,
        vocab,
        img_transforms
    )

