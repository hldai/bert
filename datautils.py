import numpy as np
import tokenization


def get_sent_tokens(tok_texts_file, vocab_file):
    # print(vocab_file)
    tokenizer = tokenization.SpaceTokenizer(vocab_file)
    f = open(tok_texts_file, encoding='utf-8')
    token_seqs = list()
    for line in f:
        raw_tokens = tokenizer.tokenize(line)
        tokens = ["[CLS]"]
        for token in raw_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        token_seqs.append(tokens)
    return token_seqs


def count_hit(terms_true, terms_pred):
    terms_true, terms_pred = terms_true.copy(), terms_pred.copy()
    terms_true.sort()
    terms_pred.sort()
    idx_pred = 0
    cnt_hit = 0
    for t in terms_true:
        while idx_pred < len(terms_pred) and terms_pred[idx_pred] < t:
            idx_pred += 1
        if idx_pred == len(terms_pred):
            continue
        if terms_pred[idx_pred] == t:
            cnt_hit += 1
            idx_pred += 1
    return cnt_hit


def get_terms_from_label_list(labels, words, label_beg, label_in):
    terms = list()
    # words = tok_text.split(' ')
    # print(labels_pred)
    # print(len(words), len(labels_pred))
    assert len(words) == len(labels)

    p = 0
    while p < len(words):
        yi = labels[p]
        if yi == label_beg:
            pright = p
            while pright + 1 < len(words) and labels[pright + 1] == label_in:
                pright += 1
            terms.append(' '.join(words[p: pright + 1]))
            p = pright + 1
        else:
            p += 1
    return terms


def prf1_for_terms(preds, token_seqs, aspect_terms_true_list, opinion_terms_true_list):
    aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
    opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
    for i, (p_seq, token_seq) in enumerate(zip(preds, token_seqs)):
        seq_len = len(token_seq)
        aspect_terms_sys = get_terms_from_label_list(p_seq[1:seq_len - 1], token_seq[1:seq_len - 1], 1, 2)
        opinion_terms_sys = get_terms_from_label_list(p_seq[1:seq_len - 1], token_seq[1:seq_len - 1], 3, 4)
        aspect_terms_true, opinion_terms_true = aspect_terms_true_list[i], opinion_terms_true_list[i]
        aspect_sys_cnt += len(aspect_terms_sys)
        aspect_true_cnt += len(aspect_terms_true)
        opinion_sys_cnt += len(opinion_terms_sys)
        opinion_true_cnt += len(opinion_terms_true)

        new_hit_cnt = count_hit(aspect_terms_true, aspect_terms_sys)
        aspect_hit_cnt += new_hit_cnt
        new_hit_cnt = count_hit(opinion_terms_true, opinion_terms_sys)
        opinion_hit_cnt += new_hit_cnt

    def prf1(n_true, n_sys, n_hit):
        p = n_hit / (n_sys + 1e-6)
        r = n_hit / (n_true + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        return p, r, f1

    aspect_p, aspect_r, aspect_f1 = prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)
    opinion_p, opinion_r, opinion_f1 = prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)
    # tf.logging.info('p={:.4f}, r={:.4f}, a_f1={:.4f}; p={:.4f}, r={:.4f}, o_f1={:.4f}'.format(
    #                 aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
    #                 opinion_f1))
    return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1


def __find_sub_words_seq(words, sub_words):
    i, li, lj = 0, len(words), len(sub_words)
    while i + lj <= li:
        j = 0
        while j < lj:
            if words[i + j] != sub_words[j]:
                break
            j += 1
        if j == lj:
            return i
        i += 1
    return -1


def __label_words_with_terms(words, terms, label_val_beg, label_val_in, x):
    for term in terms:
        term_words = term.lower().split(' ')
        pbeg = __find_sub_words_seq(words, term_words)
        if pbeg == -1:
            # print(words)
            # print(terms)
            # print()
            continue
        x[pbeg] = label_val_beg
        for p in range(pbeg + 1, pbeg + len(term_words)):
            x[p] = label_val_in


def label_sentence(words, aspect_terms=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_terms is not None:
        __label_words_with_terms(words, aspect_terms, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return seq_list


# def get_label_seqs(sents, word_seqs):
#     len_max = max([len(words) for words in word_seqs])
#     print('max sentence len:', len_max)
#
#     labels_list = list()
#     for sent_idx, (sent, sent_words) in enumerate(zip(sents, word_seqs)):
#         aspect_objs = sent.get('terms', None)
#         aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()
#
#         opinion_terms = sent.get('opinions', list())
#
#         x = label_sentence(sent_words, aspect_terms, opinion_terms)
#         labels_list.append(x)
#
#     return labels_list


def load_sents(sents_file):
    import json
    sents = list()
    with open(sents_file, encoding='utf-8') as f:
        for line in f:
            sents.append(json.loads(line))
    return sents


def get_true_terms(sents):
    aspect_terms_true_list = list()
    opinion_terms_true_list = list()
    for sent in sents:
        if aspect_terms_true_list is not None:
            aspect_terms_true_list.append(
                [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
        if opinion_terms_true_list is not None:
            opinion_terms_true_list.append([w.lower() for w in sent.get('opinions', list())])
    return aspect_terms_true_list, opinion_terms_true_list


def load_train_valid_split_labels(train_valid_split_file):
    with open(train_valid_split_file) as f:
        vals = f.read().strip().split(' ')
    return [int(v) for v in vals]


def convert_single_example(ex_index, sent, tok_sent_text, max_seq_length, tokenizer):
    import tensorflow as tf

    sent_tokens = tokenizer.tokenize(tok_sent_text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(sent_tokens) > max_seq_length - 2:
        sent_tokens = sent_tokens[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in sent_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    aspect_objs = sent.get('terms', None)
    aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()
    opinion_terms = sent.get('opinions', list())
    label_seq = label_sentence(tokens, aspect_terms, opinion_terms)
    label_seq = list(label_seq)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_seq.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_seq) == max_seq_length

    # label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        # tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    # feature = InputFeatures(
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     segment_ids=segment_ids,
    #     label_id=label_id)
    return input_ids, input_mask, segment_ids, label_seq
