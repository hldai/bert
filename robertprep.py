import tensorflow as tf
import os
import numpy as np
import collections
import json
import datautils
import config
import tokenization


def __load_json_objs(filename):
    objs = list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs


def __get_feature_dict(input_ids, input_mask, segment_ids, label_seq, seq_len):
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature(label_seq)
    features["seq_len"] = create_int_feature([seq_len])
    return features


def __gen_tf_records(tokenizer, sents, tok_texts, output_file, tokens_output_file):
    fout = open(tokens_output_file, 'w', encoding='utf-8')
    writer = tf.python_io.TFRecordWriter(output_file)
    for i, (sent, tok_text) in enumerate(zip(sents, tok_texts)):
        aspect_objs = sent.get('terms', None)
        aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()
        opinion_terms = sent.get('opinions', list())
        feats_tup = datautils.convert_single_example(
            i, tok_text, aspect_terms, opinion_terms, config.MAX_SEQ_LEN, tokenizer)
        input_ids, input_mask, segment_ids, label_seq, tokens = feats_tup

        features = __get_feature_dict(input_ids, input_mask, segment_ids, label_seq, len(tokens))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        fout.write('{}\n'.format(' '.join(tokens)))
    writer.close()
    fout.close()


def gen_test_tf_records(vocab_file, sents_file, tok_texts_file, output_file, tokens_output_file):
    tokenizer = tokenization.SpaceTokenizer(vocab_file)
    with open(sents_file, encoding='utf-8') as f:
        sents = [json.loads(line) for line in f]
    with open(tok_texts_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]
    __gen_tf_records(tokenizer, sents, tok_texts, output_file, tokens_output_file)


def gen_train_valid_tf_records(
        vocab_file, sents_file, tok_texts_file, train_valid_split_file,
        train_output_file, valid_output_file,
        train_tokens_output_file, valid_tokens_output_file):
    tokenizer = tokenization.SpaceTokenizer(vocab_file)
    with open(sents_file, encoding='utf-8') as f:
        sents = [json.loads(line) for line in f]
    with open(tok_texts_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]

    tvs_arr = datautils.load_train_valid_split_labels(train_valid_split_file)

    assert len(tvs_arr) == len(sents)

    sents_train, sents_valid = list(), list()
    tok_texts_train, tok_texts_valid = list(), list()
    for i, v in enumerate(tvs_arr):
        if v == 0:
            sents_train.append(sents[i])
            tok_texts_train.append(tok_texts[i])
        else:
            sents_valid.append(sents[i])
            tok_texts_valid.append(tok_texts[i])

    tf.logging.info('{} train, {} valid'.format(len(sents_train), len(sents_valid)))
    __gen_tf_records(tokenizer, sents_train, tok_texts_train, train_output_file, train_tokens_output_file)
    __gen_tf_records(tokenizer, sents_valid, tok_texts_valid, valid_output_file, valid_tokens_output_file)


def gen_pretrain_tfrecords(
        vocab_file, tok_texts_file, aspect_terms_file, opinion_terms_file, train_valid_idxs_file,
        train_aspect_output_file, train_opinion_output_file, valid_aspect_output_file, valid_opinion_output_file,
        valid_token_output_file):
    tokenizer = tokenization.SpaceTokenizer(vocab_file)
    aspect_terms_list = __load_json_objs(aspect_terms_file)
    opinion_terms_list = __load_json_objs(opinion_terms_file)
    tok_texts = datautils.read_lines(tok_texts_file)

    idxs_train, idxs_valid = datautils.load_train_valid_idxs(train_valid_idxs_file)
    assert len(tok_texts) == len(idxs_train) + len(idxs_valid)
    assert len(tok_texts) == len(aspect_terms_list)
    assert len(tok_texts) == len(opinion_terms_list)

    def write_records(example_idxs, aspect_output_file, opinion_output_file, token_output_file):
        writer_a = tf.python_io.TFRecordWriter(aspect_output_file)
        writer_o = tf.python_io.TFRecordWriter(opinion_output_file)
        fout = open(token_output_file, 'w', encoding='utf-8') if token_output_file else None
        for i, idx in enumerate(example_idxs):
            if (i + 1) % 10000 == 0:
                print(i + 1, len(example_idxs))
                # break

            tokens = datautils.get_sent_tokens(tok_texts[idx], tokenizer, config.MAX_SEQ_LEN)
            tmp_input_ids = tokenizer.convert_tokens_to_ids(tokens)
            tmp_aspect_label_seq = list(datautils.label_sentence(tokens, aspect_terms_list[idx]))

            (input_ids, input_mask, segment_ids, aspect_label_seq, tokens
             ) = datautils.example_to_feats(tokens, tmp_aspect_label_seq, tmp_input_ids, config.MAX_SEQ_LEN)

            features = __get_feature_dict(input_ids, input_mask, segment_ids, aspect_label_seq, len(tokens))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer_a.write(tf_example.SerializeToString())

            tmp_opinion_label_seq = list(datautils.label_sentence(tokens, opinion_terms_list[idx]))

            (input_ids, input_mask, segment_ids, opinion_label_seq, tokens
             ) = datautils.example_to_feats(tokens, tmp_opinion_label_seq, tmp_input_ids, config.MAX_SEQ_LEN)

            features = __get_feature_dict(input_ids, input_mask, segment_ids, opinion_label_seq, len(tokens))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer_o.write(tf_example.SerializeToString())

            # (input_ids, input_mask, segment_ids, label_seq, tokens
            #  ) = datautils.convert_single_example_single_term_type(
            #     i, tok_texts[idx], terms_list[idx], config.MAX_SEQ_LEN, tokenizer)
            # print(input_ids)
            # print(label_seq)
            # print(tokens)
            # exit()
            if fout is not None:
                fout.write('{}\n'.format(' '.join(tokens)))
        writer_a.close()
        writer_o.close()
        if fout is not None:
            fout.close()

    write_records(idxs_train, train_aspect_output_file, train_opinion_output_file, None)
    write_records(idxs_valid, valid_aspect_output_file, valid_opinion_output_file, valid_token_output_file)


if __name__ == '__main__':
    dataset = 'se14l'
    # dataset = 'se14r'
    # dataset = 'se15r'
    yelp_pretrain_tok_texts_file = os.path.join(
        config.RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-part0_04.txt')
    yelp_tv_idxs_file = os.path.join(config.RES_DIR, 'yelp/eng-part/yelp-rest-sents-r9-tok-eng-part0_04-tvidxs.txt')
    amazon_pretrain_tok_texts_file = os.path.join(
        config.RES_DIR, 'amazon/laptops-reivews-sent-tok-text.txt')
    amazon_tv_idxs_file = os.path.join(config.RES_DIR, 'amazon/laptops-reivews-sent-tok-text-tvidxs.txt')

    dataset_files = config.DATASET_FILES[dataset]
    pretrain_tok_texts_file = amazon_pretrain_tok_texts_file if dataset == 'se14l' else yelp_pretrain_tok_texts_file
    tv_idxs_file = amazon_tv_idxs_file if dataset == 'se14l' else yelp_tv_idxs_file

    # gen_train_valid_tf_records(
    #     config.BERT_VOCAB_FILE,
    #     dataset_files['train_sents_file'], dataset_files['train_tok_texts_file'],
    #     dataset_files['train_valid_split_file'], dataset_files['train_tfrecord_file'],
    #     dataset_files['valid_tfrecord_file'], dataset_files['bert_train_tokens_file'],
    #     dataset_files['bert_valid_tokens_file']
    # )
    # gen_test_tf_records(
    #     config.BERT_VOCAB_FILE,
    #     dataset_files['test_sents_file'], dataset_files['test_tok_texts_file'],
    #     dataset_files['test_tfrecord_file'], dataset_files['bert_test_tokens_file'])
    # gen_pretrain_tfrecords(
    #     vocab_file=config.BERT_VOCAB_FILE, tok_texts_file=pretrain_tok_texts_file,
    #     aspect_terms_file=dataset_files['pretrain_aspect_terms_file'],
    #     opinion_terms_file=dataset_files['pretrain_opinion_terms_file'], train_valid_idxs_file=tv_idxs_file,
    #     train_aspect_output_file=dataset_files['pretrain_train_aspect_tfrec_file'],
    #     train_opinion_output_file=dataset_files['pretrain_train_opinion_tfrec_file'],
    #     valid_aspect_output_file=dataset_files['pretrain_valid_aspect_tfrec_file'],
    #     valid_opinion_output_file=dataset_files['pretrain_valid_opinion_tfrec_file'],
    #     valid_token_output_file=dataset_files['pretrain_valid_token_file']
    # )
