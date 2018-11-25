BERT_BASE_DIR="/home/hldai/data/bert/uncased_L-12_H-768_A-12"
DST_DIR="/home/hldai/data/bert"

python create_pretraining_data.py \
  --input_file=/home/hldai/data/yelp/yelp-review-eng-tok-sents-round-9.txt \
  --output_file=$DST_DIR/yelp-bert.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
