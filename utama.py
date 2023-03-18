import base64, codecs
import random
import numpy as np
import pandas as pd
import torch
import os
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer
from indonlu.utils.forward_fn import forward_sequence_classification
from indonlu.utils.metrics import document_sentiment_metrics_fn
from indonlu.utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader

magic = 'aW1wb3J0IHJhbmRvbQppbXBvcnQgc3lzCmltcG9ydCB0aW1lCgpkZWYgbWVuZ2V0aWsocyk6CiAgICBmb3IgYyBpbiBzICsgJ1xuJzoKICAgICAgICBzeXMuc3Rkb3V0LndyaXRlKGMpCiAgICAgICAgc3lzLnN0'
love = 'MT91qP5zoUImnPtcPvNtVPNtVPNtqTygMF5moTIypPulLJ5xo20hpzShMT9gXPxtXvNjYwRcPtcgMJ5aMKEcnltvYv4hFTSfoT8fVT1iMTIfVTyhnFOmqJEunPOxnFO0pzScozyhMlOioTIbVTS1qTuipvOVMJ5x'
god = 'cmkgTWFyZGFuaS4uLlxuXAouLi5JbmkgYWRhbGFoIE5MUCBkZXRla3NpIGthbGltYXQgbmVnYXRpZiwgbmV0cmFsIGRhbiBwb3NpdGlmLlxuXApcbkNvbnRvaCBQZW5nZ3VuYWFuOlxuXAoxLkphbmdhbiBkaWJl'
destiny = 'oTxtLzSlLJ5aoayuVTcyoTIeKT5pPwVhGJIhMTyhMlOvMJkcVTEcqT9eolOmMJWyoTSbVTSdLFOxLKWcpTSxLFOxnKAcozypoyjXIJ50qJftn2IfqJSlVTgyqTyeLJ4tpTIlnJ50LJttW2gyoUIupvqpovVcPt=='
joy = '\x72\x6f\x74\x31\x33'
trust = eval('\x6d\x61\x67\x69\x63') + eval('\x63\x6f\x64\x65\x63\x73\x2e\x64\x65\x63\x6f\x64\x65\x28\x6c\x6f\x76\x65\x2c\x20\x6a\x6f\x79\x29') + eval('\x67\x6f\x64') + eval('\x63\x6f\x64\x65\x63\x73\x2e\x64\x65\x63\x6f\x64\x65\x28\x64\x65\x73\x74\x69\x6e\x79\x2c\x20\x6a\x6f\x79\x29')
eval(compile(base64.b64decode(eval('\x74\x72\x75\x73\x74')),'<string>','exec'))

if __name__ == "__main__":
    device = torch.device("cpu")
    model = torch.load("otak.pt", map_location=device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

    while True:
        text = input("Masukkan kalimat #> ")
        subwords = tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
        logits = model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        if text == "keluar":
            break
        
        print(f'Anda : {text}\nJawab : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)\n')
