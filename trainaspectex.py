import tensorflow as tf
import os
import config
import json
import collections
import tokenization
import datautils
import modeling
import optimization
from platform import platform


def __gen_tf_records(tokenizer, sents, tok_texts, output_file):
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    writer = tf.python_io.TFRecordWriter(output_file)
    for i, (sent, tok_text) in enumerate(zip(sents, tok_texts)):
        feats_tup = datautils.convert_single_example(i, sent, tok_text, max_seq_len, tokenizer)
        input_ids, input_mask, segment_ids, label_seq = feats_tup

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["label_ids"] = create_int_feature(label_seq)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def __gen_test_tf_records(sents_file, tok_texts_file, output_file, vocab_file):
    tokenizer = tokenization.SpaceTokenizer(vocab_file)
    with open(sents_file, encoding='utf-8') as f:
        sents = [json.loads(line) for line in f]
    with open(tok_texts_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]
    __gen_tf_records(tokenizer, sents, tok_texts, output_file)


def __gen_train_valid_tf_records(sents_file, tok_texts_file, train_valid_split_file,
                                 train_output_file, valid_output_file, vocab_file):
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
    __gen_tf_records(tokenizer, sents_train, tok_texts_train, train_output_file)
    __gen_tf_records(tokenizer, sents_valid, tok_texts_valid, valid_output_file)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value
    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, max_seq_len, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        per_example_loss = tf.dtypes.cast(input_mask, tf.float32) * per_example_loss
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return loss, per_example_loss, logits, predict


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels)

        # use checkpoint
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # logging out trainable variables
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    # "per_example_loss": per_example_loss
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def __get_num_sents(sents_file):
    n_sents = 0
    with open(sents_file) as f:
        for _ in f:
            n_sents += 1
    return n_sents


def __get_num_train_valid_samples(train_valid_split_file):
    vals = datautils.load_train_valid_split_labels(train_valid_split_file)
    n_train_samples = 0
    for v in vals:
        if v == 0:
            n_train_samples += 1
    return n_train_samples, len(vals) - n_train_samples


def __run_aspectex_bert(
        train_file, init_checkpoint, learning_rate, train_valid_split_file, output_dir,
        valid_file=None, eval_file=None, train_sents_file=None, train_tok_texts_file=None, test_sents_file=None,
        test_tok_texts_file=None, vocab_file=None):
    tf.logging.info(train_file)

    bert_config = modeling.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            per_host_input_for_training=is_per_host))

    # n_train_examples = __get_num_sents(train_sents_file)
    n_train_examples, n_valid_examples = __get_num_train_valid_samples(train_valid_split_file)
    num_train_steps = int(
        n_train_examples / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    tf.logging.info('{} train steps'.format(num_train_steps))
    tf.logging.info('{} train steps per epoch'.format(int(n_train_examples / train_batch_size)))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=n_labels,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    if test_sents_file is None:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", n_train_examples)
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_len,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if test_sents_file is not None:
        valid_input_fn = file_based_input_fn_builder(
            input_file=valid_file,
            seq_length=max_seq_len,
            is_training=False,
            drop_remainder=False)
        tf.logging.info(valid_file)
        preds = estimator.predict(valid_input_fn)
        preds = [y for y in preds]
        train_valid_split_labels = datautils.load_train_valid_split_labels(train_valid_split_file)
        token_seqs = datautils.get_sent_token_seqs(train_tok_texts_file, vocab_file)
        valid_token_seqs = [token_seq for token_seq, tmpl in zip(token_seqs, train_valid_split_labels) if tmpl == 1]
        assert len(preds) == len(valid_token_seqs)
        all_train_sents = datautils.load_sents(train_sents_file)
        valid_sents = [sent for sent, tmpl in zip(all_train_sents, train_valid_split_labels) if tmpl == 1]
        assert len(preds) == len(valid_sents)
        aspect_terms_true_list, opinion_terms_true_list = datautils.get_true_terms(valid_sents)
        (a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v
         ) = datautils.prf1_for_terms(preds, valid_token_seqs, aspect_terms_true_list, opinion_terms_true_list)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_len,
            is_training=False,
            drop_remainder=False)
        tf.logging.info(eval_file)
        # result = estimator.evaluate(input_fn=eval_input_fn, steps=None)
        # tf.logging.info('loss={}, acc={}'.format(result['eval_loss'], result['eval_accuracy']))

        preds = estimator.predict(eval_input_fn)
        token_seqs = datautils.get_sent_token_seqs(test_tok_texts_file, vocab_file)
        test_sents = datautils.load_sents(test_sents_file)
        aspect_terms_true_list, opinion_terms_true_list = datautils.get_true_terms(test_sents)
        (a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t
         ) = datautils.prf1_for_terms(preds, token_seqs, aspect_terms_true_list, opinion_terms_true_list)
        tf.logging.info(init_checkpoint)
        f1_sum_valid = a_f1_v + o_f1_v
        tf.logging.info(
            'Valid, p={:.4f}, r={:.4f}, a_f1={:.4f}; p={:.4f}, r={:.4f}, o_f1={:.4f}, f1_sum={:.4f}'.format(
                a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v, f1_sum_valid))
        tf.logging.info('Test, p={:.4f}, r={:.4f}, a_f1={:.4f}; p={:.4f}, r={:.4f}, o_f1={:.4f}'.format(
            a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # dataset = 'se14r'
    dataset = 'se15r'
    iterations_per_loop = 1000
    eval_batch_size, predict_batch_size = 8, 8
    warmup_proportion = 0.1
    num_tpu_cores = 1
    save_checkpoints_steps = 200
    train_batch_size = 32
    num_train_epochs = 50.0
    max_seq_len = 128
    n_labels = 5
    learning_rate = 5e-5
    if dataset == 'se14r':
        init_checkpoint_for_test = os.path.join(config.SE14_DIR, 'restaurants/bert-output/model.ckpt-2800')
    elif dataset == 'se15r':
        init_checkpoint_for_test = os.path.join(config.SE15_DIR, 'restaurants/bert-output/model.ckpt-800')
    else:
        init_checkpoint_for_test = os.path.join(config.SE14_DIR, 'laptops/bert-output/model.ckpt-2800')

    dataset_files = config.DATASET_FILES[dataset]

    __run_aspectex_bert(
        dataset_files['train_tfrecord_file'], init_checkpoint_for_test, learning_rate,
        dataset_files['train_valid_split_file'], dataset_files['bert_output_dir'],
        dataset_files['valid_tfrecord_file'], dataset_files['test_tfrecord_file'],
        dataset_files['train_sents_file'], dataset_files['train_tok_texts_file'],
        dataset_files['test_sents_file'], dataset_files['test_tok_texts_file'], config.BERT_VOCAB_FILE)
