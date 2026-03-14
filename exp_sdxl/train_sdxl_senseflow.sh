work_path=$(dirname $0)
filename=$(basename $work_path)
T=$(date +%m%d%H%M)
OMP_NUM_THREADS=1 \
PYTHONFAULTHANDLER=True \
torchrun \
--nproc_per_node $2 \
--nnodes $1 \
main_trainer_sdxl_senseflow.py --config_path $3 --save_path $4
