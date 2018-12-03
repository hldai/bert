from platform import platform
from os.path import join

NUM_LABELS = 5
MAX_SEQ_LEN = 128

os_env = 'Windows' if platform().startswith('Windows') else 'Linux'


if os_env == 'Windows':
    BERT_BASE_DIR = 'd:/data/res/bert'
    RES_DIR = 'd:/data/res/'
    SE14_DIR = 'd:/data/aspect/semeval14'
    SE15_DIR = 'd:/data/aspect/semeval15'
else:
    BERT_BASE_DIR = '/home/hldai/data/bert'
    RES_DIR = '/home/hldai/data/res/'
    SE14_DIR = '/home/hldai/data/aspect/semeval14'
    SE15_DIR = '/home/hldai/data/aspect/semeval15'

BERT_CONFIG_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/bert_config.json')
BERT_VOCAB_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/vocab.txt')

SE14L_FILES = {
    'train_sents_file': join(SE14_DIR, 'laptops/laptops_train_sents.json'),
    'test_sents_file': join(SE14_DIR, 'laptops/laptops_test_sents.json'),
    'train_valid_split_file': join(SE14_DIR, 'laptops/laptops_train_valid_split.txt'),
    'train_tok_texts_file': join(SE14_DIR, 'laptops/laptops_train_texts_tok.txt'),
    'test_tok_texts_file': join(SE14_DIR, 'laptops/laptops_test_texts_tok.txt'),
    'bert_output_dir': join(SE14_DIR, 'laptops/bert-output'),
    'train_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-train.tfrecord'),
    'valid_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-valid.tfrecord'),
    'test_tfrecord_file': join(SE14_DIR, 'bert-data/se14l-test.tfrecord'),
    'bert_train_tokens_file': join(SE14_DIR, 'bert-data/se14l-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE14_DIR, 'bert-data/se14l-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE14_DIR, 'bert-data/se14l-test-tokens.txt'),
    'init_checkpoint': join(BERT_BASE_DIR, 'amazon/model.ckpt-10000'),
    'pretrain_aspect_terms_file': join(SE14_DIR, 'laptops/amazon-laptops-aspect-rm-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE14_DIR, 'laptops/amazon-laptops-aspect-rm-rule-result.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-train-aspect.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-aspect.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-train-opinion.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-opinion.tfrecord'),
    'pretrain_valid_token_file': join(SE14_DIR, 'laptops/se14l-amazonlaptops-valid-tokens.txt'),
}

SE14R_FILES = {
    'train_sents_file': join(SE14_DIR, 'restaurants/restaurants_train_sents.json'),
    'test_sents_file': join(SE14_DIR, 'restaurants/restaurants_test_sents.json'),
    'train_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_train_texts_tok.txt'),
    'test_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_test_texts_tok.txt'),
    'train_valid_split_file': join(SE14_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'bert_output_dir': join(SE14_DIR, 'restaurants/bert-output'),
    'train_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-train.tfrecord'),
    'valid_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-valid.tfrecord'),
    'test_tfrecord_file': join(SE14_DIR, 'bert-data/se14r-test.tfrecord'),
    'bert_train_tokens_file': join(SE14_DIR, 'bert-data/se14r-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE14_DIR, 'bert-data/se14r-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE14_DIR, 'bert-data/se14r-test-tokens.txt'),
    'init_checkpoint': join(BERT_BASE_DIR, 'yelp/model.ckpt-10000'),
    'pretrain_aspect_terms_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-aspect-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-opinion-rule-result.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-train-aspect.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-aspect.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-train-opinion.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-opinion.tfrecord'),
    'pretrain_valid_token_file': join(SE14_DIR, 'restaurants/se14r-yelpr9-rest-p0_04-valid-tokens.txt'),
}

SE15R_FILES = {
    'train_sents_file': join(SE15_DIR, 'restaurants/restaurants_train_sents.json'),
    'test_sents_file': join(SE15_DIR, 'restaurants/restaurants_test_sents.json'),
    'train_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_train_texts_tok.txt'),
    'test_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_test_texts_tok.txt'),
    'train_valid_split_file': join(SE15_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'bert_output_dir': join(SE15_DIR, 'restaurants/bert-output'),
    'train_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-train.tfrecord'),
    'valid_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-valid.tfrecord'),
    'test_tfrecord_file': join(SE15_DIR, 'bert-data/se15r-test.tfrecord'),
    'bert_train_tokens_file': join(SE15_DIR, 'bert-data/se15r-train-tokens.txt'),
    'bert_valid_tokens_file': join(SE15_DIR, 'bert-data/se15r-valid-tokens.txt'),
    'bert_test_tokens_file': join(SE15_DIR, 'bert-data/se15r-test-tokens.txt'),
    'init_checkpoint': join(BERT_BASE_DIR, 'yelp/model.ckpt-10000'),
    'pretrain_aspect_terms_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-aspect-rule-result.txt'),
    'pretrain_opinion_terms_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-opinion-rule-result.txt'),
    'pretrain_train_aspect_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-train-aspect.tfrecord'),
    'pretrain_valid_aspect_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-aspect.tfrecord'),
    'pretrain_train_opinion_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-train-opinion.tfrecord'),
    'pretrain_valid_opinion_tfrec_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-opinion.tfrecord'),
    'pretrain_valid_token_file': join(SE15_DIR, 'restaurants/yelpr9-rest-p0_04-valid-tokens.txt'),
}

DATASET_FILES = {
    'se14l': SE14L_FILES,
    'se14r': SE14R_FILES,
    'se15r': SE15R_FILES
}
