import tensorflow as tf
import os
import config
import json
import collections
import tokenization
import datautils
import dhlmodeling
import optimization
from platform import platform


def __get_dataset(data_file, batch_size, is_train, seq_length):
    dataset = tf.data.TFRecordDataset(data_file)
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

    if is_train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=100)

    drop_remainder = True if is_train else False
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return dataset


def model_init(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels,
        init_checkpoint, num_train_steps, num_warmup_steps, hidden_dropout, attention_dropout):
    model = dhlmodeling.BertModel(
        config=bert_config,
        input_ids=input_ids,
        hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout,
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

    # use checkpoint
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = dhlmodeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    train_op = optimization.create_optimizer(
        loss, learning_rate, num_train_steps, num_warmup_steps, False)
    return train_op, loss, logits, predict


def __evaluate(dataset_valid, sess, predict, input_ids, input_mask, segment_ids,
               label_ids, hidden_dropout, attention_probs_dropout, token_seqs, at_true_list, ot_true_list):
    next_valid_example = dataset_valid.make_one_shot_iterator().get_next()
    all_preds = list()
    idx = 0
    while True:
        try:
            features = sess.run(next_valid_example)
            predict_vals = sess.run(predict, feed_dict={
                input_ids: features["input_ids"], input_mask: features["input_mask"],
                segment_ids: features["segment_ids"], label_ids: features["label_ids"],
                hidden_dropout: 1.0, attention_probs_dropout: 1.0
            })
            if idx == 7:
                print(features['input_ids'])
                print(token_seqs[:eval_batch_size])
                print()
            idx += 1
            for y_pred in predict_vals:
                all_preds.append(y_pred)
        except tf.errors.OutOfRangeError:
            break
    (a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v
     ) = datautils.prf1_for_terms(all_preds, token_seqs, at_true_list, ot_true_list)
    f1_sum = a_f1_v + o_f1_v
    tf.logging.info(
        'Valid, p={:.4f}, r={:.4f}, a_f1={:.4f}; p={:.4f}, r={:.4f}, o_f1={:.4f}, f1_sum={:.4f}'.format(
            a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v, f1_sum))
    return f1_sum


def __train_robert(
        vocab_file, train_sents_file, train_tok_texts_file, train_valid_split_file,
        test_sents_file, test_tok_texts_file, train_data_file, valid_data_file, test_data_file, batch_size,
        seq_length, init_checkpoint, num_epochs):
    bert_config = dhlmodeling.BertConfig.from_json_file(config.BERT_CONFIG_FILE)

    train_valid_split_labels = datautils.load_train_valid_split_labels(train_valid_split_file)
    all_train_sents = datautils.load_sents(train_sents_file)
    valid_sents = [sent for sent, tmpl in zip(all_train_sents, train_valid_split_labels) if tmpl == 1]

    dataset_train = __get_dataset(train_data_file, batch_size, True, seq_length)
    next_train_example = dataset_train.make_one_shot_iterator().get_next()

    dataset_valid = __get_dataset(valid_data_file, batch_size, False, seq_length)
    dataset_test = __get_dataset(test_data_file, batch_size, False, seq_length)

    token_seqs = datautils.get_sent_tokens(train_tok_texts_file, vocab_file)
    valid_token_seqs = [token_seq for token_seq, tmpl in zip(token_seqs, train_valid_split_labels) if tmpl == 1]
    valid_at_true_list, valid_ot_true_list = datautils.get_true_terms(valid_sents)

    test_token_seqs = datautils.get_sent_tokens(test_tok_texts_file, vocab_file)
    test_sents = datautils.load_sents(test_sents_file)
    test_at_true_list, test_ot_true_list = datautils.get_true_terms(test_sents)

    n_train_examples = len(all_train_sents) - len(valid_sents)
    n_train_steps_per_epoch = int(n_train_examples / train_batch_size)
    num_train_steps = n_train_steps_per_epoch * num_epochs
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    hidden_dropout = tf.placeholder(tf.float32, shape=[], name="hidden_dropout_prob")
    attention_probs_dropout = tf.placeholder(
        tf.float32, shape=[], name="attention_probs_dropout_prob")

    input_ids = tf.placeholder(tf.int32, [None, seq_length])
    input_mask = tf.placeholder(tf.int32, [None, seq_length])
    segment_ids = tf.placeholder(tf.int32, [None, seq_length])
    label_ids = tf.placeholder(tf.int32, [None, seq_length])

    train_op, loss, logits, predict = model_init(
        bert_config=bert_config, is_training=True, input_ids=input_ids,
        input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids,
        num_labels=config.NUM_LABELS, init_checkpoint=init_checkpoint,
        num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, hidden_dropout=hidden_dropout,
        attention_dropout=attention_probs_dropout
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = list()
    for step in range(num_train_steps):
        features = sess.run(next_train_example)
        # input_ids = features["input_ids"]
        # input_mask = features["input_mask"]
        # segment_ids = features["segment_ids"]
        # label_ids = features["label_ids"]
        _, loss_val = sess.run([train_op, loss], feed_dict={
            input_ids: features["input_ids"], input_mask: features["input_mask"],
            segment_ids: features["segment_ids"], label_ids: features["label_ids"],
            hidden_dropout: 0.9, attention_probs_dropout: 0.9
        })
        losses.append(loss_val)

        if (step + 1) % n_train_steps_per_epoch == 0:
            epoch = int((step + 1) / n_train_steps_per_epoch)
            tf.logging.info('{} {} {}'.format(epoch, step + 1, sum(losses) / len(losses)))
            losses = list()
            f1_sum_valid = __evaluate(
                dataset_valid, sess, predict, input_ids, input_mask, segment_ids, label_ids, hidden_dropout,
                attention_probs_dropout, valid_token_seqs, valid_at_true_list, valid_ot_true_list)
            f1_sum_valid = __evaluate(
                dataset_test, sess, predict, input_ids, input_mask, segment_ids, label_ids, hidden_dropout,
                attention_probs_dropout, test_token_seqs, test_at_true_list, test_ot_true_list)


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
    max_seq_len = 128
    n_labels = 5
    learning_rate = 5e-5

    dataset_files = config.DATASET_FILES[dataset]
    seq_length = 128
    batch_size = 5
    n_epochs = 50

    __train_robert(
        config.BERT_VOCAB_FILE, dataset_files['train_sents_file'], dataset_files['train_tok_texts_file'],
        dataset_files['train_valid_split_file'], dataset_files['test_sents_file'],
        dataset_files['test_tok_texts_file'],
        dataset_files['train_tfrecord_file'], dataset_files['valid_tfrecord_file'],
        dataset_files['test_tfrecord_file'],
        batch_size, seq_length, dataset_files['init_checkpoint'], n_epochs)
