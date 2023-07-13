DATA_SEEDS=$(seq 10)
MODEL_SEEDS=$(seq 3)
for DATA_SEED in $DATA_SEEDS; do
    for MODEL_SEED in $MODEL_SEEDS; do
        RANDOM=$MODEL_SEED
        echo
        echo Running seed $DATA_SEED
        python experimental_design.py --data_seed $DATA_SEED --seed $MODEL_SEED "$@";
    done
done
