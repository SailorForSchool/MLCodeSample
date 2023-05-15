'''
This file contains language utility functions for both the encoder-decoder model and
the auto-regressive model. Any function for the autoregressive model is followed by
a '_r'.
'''

import torch
import spacy
from torchtext.data.metrics import bleu_score
from tqdm.notebook import tqdm_notebook
import sys
import utils

def translate_sentence(model, sentence, english, dutch, device, max_length=50, top_k=0, top_p=.75, temp=0.5):
    
    # Load english tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_eng(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Go through each english token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]
    
    if len(text_to_indices) > 80:
        text_to_indices = text_to_indices[:80]
    
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).transpose(0,1).to(device)

    # get encoder hidden states
    with torch.no_grad():
        src_mask = model.make_src_mask(sentence_tensor)
        enc_src = model.encoder(sentence_tensor, src_mask)

    outputs = [dutch.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_words = torch.LongTensor(outputs).unsqueeze(1).transpose(0,1).to(device)

        with torch.no_grad():
            output = model.decoder(previous_words, enc_src, src_mask,
                                   model.make_causal_mask(previous_words))
        
            output = output[0, -1, :] / temp
            output = utils.top_k_top_p_filtering(output, top_k, top_p)
            output = output.softmax(dim=0)

        # sample happens here
        best_guess = torch.multinomial(output, 1)
        outputs.append(best_guess.item())

        # Model predicts it's the end of the sentence
        if outputs[-1] == dutch.vocab.stoi["<eos>"]:
            break
            
    # free gpu space
    del sentence_tensor
    del src_mask
    del previous_words
    del output
    del best_guess

    translated_sentence = [dutch.vocab.itos[idx] for idx in outputs]
        
    return translated_sentence[1:]

def translate_sentence_r(model, sentence, english, dutch, device, max_length=50, top_k=0, top_p=.75, temp=0.5):
    
    # Load english tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_eng(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Go through each english token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]
    
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).transpose(0,1).to(device)

    outputs = [dutch.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor(outputs[-1]).unsqueeze(1).transpose(0,1).to(device)
        sentence_tensor = torch.cat((sentence_tensor, previous_word), 1)

        with torch.no_grad():
            output = model(sentence_tensor)
        
            output = output[0, -1, :] / temp
            output = utils.top_k_top_p_filtering(output, top_k, top_p)
            output = output.softmax(dim=0)

        # sample happens here
        best_guess = torch.multinomial(output, 1)
        outputs.append(best_guess.item())

        # Model predicts it's the end of the sentence
        if outputs[-1] == dutch.vocab.stoi["<eos>"]:
            break
            
    # free gpu space
    del sentence_tensor
    del previous_words
    del output
    del best_guess

    translated_sentence = [dutch.vocab.itos[idx] for idx in outputs]

    # remove start token and create string
    nl_sent = ""
    for word in translated_sentence[1:-1]:
        # todo? maybe remove space for punctuation and capitalize first words
        nl_sent = nl_sent + " " + word
        
    return nl_sent

def bleu(data, model, max_length, english, dutch, device):
    
    targets = []
    outputs = []

    for example in tqdm_notebook(data, desc = "Bleu Score Progress"):       

        src = vars(example)["English"]
        trg = vars(example)["Dutch"]        
        
        prediction = translate_sentence(model, src, english, dutch, device, max_length)
        prediction = prediction[:-1]

        targets.append([trg])
        outputs.append(prediction)                         
         
    return bleu_score(outputs, targets)

def bleu_r(data, model, max_length, english, dutch, device):
     
    targets = []
    outputs = []
    
    for example in tqdm_notebook(data, desc = "Bleu Score Progress"):       

        src = vars(example)["English"]
        trg = vars(example)["Dutch"]        
        
        prediction = translate_sentence_r(model, src, english, dutch, device, max_length)
        prediction = prediction[:-1]

        targets.append([trg])
         
    return bleu_score(outputs, targets)