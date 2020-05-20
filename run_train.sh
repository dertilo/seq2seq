#
OUTPUT_DIR_NAME=bart_seq2seq_dialogue
export OUTPUT_DIR=$HOME/data/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

cd ../transformers
export PYTHONPATH="examples/":"${PYTHONPATH}"
NUM_NODES=$1
NUM_GPUS=$2
BATCH_SIZE=$(($NUM_NODES*$NUM_GPUS))

python examples/summarization/bart/finetune.py \
--data_dir=$HOME/data/seq2seq_dialogue \
--model_name_or_path=bart-large-cnn \
--learning_rate=3e-5 \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--output_dir=$OUTPUT_DIR \
--num_nodes $NUM_NODES \
--n_gpu $NUM_GPUS \
--do_train
