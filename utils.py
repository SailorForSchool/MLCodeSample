import torch
import nltk
import pandas as pd
from text_preprocessing import text_preprocessing
from top_k import top_k_top_p_filtering

def CSVtoString(file_path, num_entries="all"):

    data = pd.read_csv(file_path)
    text = ""

    if num_entries == "all":
        num_entries = len(data)

    for row in range(num_entries):
        for column in range(len(list(data))):
            text = text + str(data.values[row][column]) + " "

    return text


def tokenizeData(data, max_length=100, device="cpu"):

    sentences = nltk.sent_tokenize(data)
    clean = []

    for sent in sentences:
        sent = "<SOS>" + sent + "<EOS>"
        clean_s =  text_preprocessing(
            sent, 
            punctuations=False, 
            lemmatization=False, 
            remove_num=False, 
            stop_words=False,
            convert_num=False
            )
        if len(clean_s) < max_length:
           for _ in range(max_length - len(clean_s)):
               clean_s.append("<pad>")
        elif len(clean_s) > max_length:
            clean_s = clean_s[:max_length]
        if len(clean_s) > 0:
            clean.append(clean_s)

    # NOTE: vocab is a dictionary
    vocab, vocab_size = getVocab(clean)

    data_dict = {
        "text" : clean,
        "vocab" : vocab,
        "vocab_size" : vocab_size + 1,
        "max_seq" : max_length,
        "pad_idx" : 0
        }

    model_input = prepare_ids(data_dict)
    model_input = torch.tensor(model_input).to(device)

    return data_dict, model_input

def getVocab(tok_text):
    vocab = {}
    vocab["<pad>"] = 0

    word_idx = 1

    for sent in tok_text:
        for word in sent:
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1

    return vocab, len(vocab)

def prepare_ids(data_dict):
    vocab = data_dict["vocab"]
    sentences = data_dict["text"]
    updated_sentences = []

    for sent in range(len(sentences)):
        u_sent = []
        for word in sentences[sent]:
            u_sent.append(vocab[word])
        updated_sentences.append(u_sent)

    return updated_sentences

def decode_sent(vocab, sentence):

    vocab_list = list(vocab)
    new_sent = ""

    for word_idx in sentence:
        if word_idx == -1:
            continue
        new_sent = new_sent + vocab_list[word_idx] + " "

    return new_sent

def sample_sentence(model, x, temp, vocab, device, max_length=100, top_k=0, top_p=.75):

    clean_s =  text_preprocessing(
        x, 
        punctuations=False, 
        lemmatization=False, 
        remove_num=False, 
        stop_words=False,
        convert_num=False
    )

    vocab_list = list(vocab)

    input = []
    for word in clean_s:
        input.append(vocab[word])

    m_input = torch.tensor(input).unsqueeze(1).to(device)
    trg = input
        
    while (len(input) < max_length):

        t_trg = torch.tensor(trg).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(m_input, t_trg)

        output = output[0, -1, :] / temp
        filtered_output = top_k_top_p_filtering(output, top_k, top_p)
        prob = filtered_output.softmax(dim=0)

        # sample happens here
        best_guess = torch.multinomial(prob, 1)
        trg.append(best_guess)

        new_word = vocab_list[best_guess]
        x = x + " " + new_word

    return x
