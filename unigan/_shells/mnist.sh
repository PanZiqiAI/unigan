
if [ "$1" == "baseline" ]; then
    python script_train.py \
        --version=v1 --dataset=mnist \
        --ncs 128 64 32 8 --hidden_ncs 128 64 32 16 --middle_ns_flows 3 3 3 \
        --disc=vanilla --ndf=64 --nz=16 \
        --trigger_disc=5 --trigger_gen=1 \
        --freq_step_gunif=-1 \
        --steps=200000 --batch_size=64 --eval_jacob_num_samples=400 \
        --desc="mnist/baseline"
elif [ "$1" == "gunif" ]; then
    TRIG_GU=0
    FREQ_STEP_GU=1
    SN_POWER=3
    LAMBDA_GU=1.0
    python script_train.py \
        --version=v1 --dataset=mnist \
        --ncs 128 64 32 8 --hidden_ncs 128 64 32 16 --middle_ns_flows 3 3 3 \
        --disc=vanilla --ndf=64 --nz=16 \
        --trigger_disc=5 --trigger_gen=1 \
        --trigger_gunif="${TRIG_GU}" --freq_step_gunif="${FREQ_STEP_GU}" --sn_power="${SN_POWER}" --lambda_gen_gunif="${LAMBDA_GU}" \
        --steps=200000 --batch_size=64 --eval_jacob_num_samples=400 \
        --desc="mnist/gu@${TRIG_GU}#${FREQ_STEP_GU}#${SN_POWER}#${LAMBDA_GU}"
fi
