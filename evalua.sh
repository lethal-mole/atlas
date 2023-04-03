DATA_DIR=./atlas_data
SIZE=large # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

 port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/train.json"
EVAL_FILES="${DATA_DIR}/dev.json"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=pruebaUse
TRAIN_STEPS=30


python evaluate.py --name 'pruebaUse' --generation_max_length 200 --gold_score_mode "pdist" --precision fp32 --reader_model_type google/t5-${SIZE}-lm-adapt --text_maxlength 100 --model_path ${DATA_DIR}/models/atlas_nq/${SIZE} --per_gpu_batch_size 1  --per_gpu_embedder_batch_size 512 --n_context 40 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --main_port $port --index_mode "flat"  --task "qa" --use_file_passages --passages "./atlas_data/usePassage.jsonl" --write_results --eval_data "./input.json"
