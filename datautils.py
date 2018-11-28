import numpy as np


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


def convert_single_example(ex_index, sent, tok_sent_text, max_seq_length, tokenizer):
    import tensorflow as tf
    import tokenization

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
