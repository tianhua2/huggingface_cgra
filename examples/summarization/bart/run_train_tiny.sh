# Install newest ptl.
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_tiny.tgz
tar -xzvf cnn_tiny.tgz


export OUTPUT_DIR_NAME=bart_utest_output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access transformer_base.py
#export PYTHONPATH="../../":"${PYTHONPATH}"
NGPU = 0
python run_bart_sum.py \
--data_dir=cnn_tiny/ \
--model_type=bart \
--model_name_or_path=sshleifer/bart-tiny-random \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=1  \
--n_gpu=0 \
--do_train $@

rm -rf cnn_tiny
rm -rf $OUTPUT_DIR



