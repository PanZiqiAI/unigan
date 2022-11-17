
if [ "$1" == "baseline" ]; then
    python script_train.py \
        --dataset=single-mnist --dataset_category=8 \
        --ncs 128 64 32 8 --hidden_ncs 128 64 32 16 --middle_ns_flows 3 3 3 \
        --disc=vanilla --ndf=64 --nz=16 \
        --trigger_disc=5 --trigger_gen=1 \
        --freq_step_gunif=-1 \
        --steps=200000 --batch_size=64 \
        --desc="single-mnist@8/baseline"
elif [ "$1" == "gunif" ]; then
    VER="$2"
    TRIG_GU=0
    FREQ_STEP_GU=1
    SN_POWER=3
    if [ "${VER}" == "v1" ]; then
        LAMBDA_GU=1.0
        python script_train.py \
            --version=v1 --dataset=single-mnist --dataset_category=8 \
            --ncs 128 64 32 8 --hidden_ncs 128 64 32 16 --middle_ns_flows 3 3 3 \
            --disc=vanilla --ndf=64 --nz=16 \
            --trigger_disc=5 --trigger_gen=1 \
            --trigger_gunif="${TRIG_GU}" --freq_step_gunif="${FREQ_STEP_GU}" --sn_power="${SN_POWER}" --lambda_gen_gunif="${LAMBDA_GU}" --loss_gunif_reduction="sum" \
            --steps=200000 --batch_size=64 \
            --desc="single-mnist@8/gu@${TRIG_GU}#${FREQ_STEP_GU}#${SN_POWER}#${LAMBDA_GU}"
    elif [ "${VER}" == "v2" ]; then
        TRIG_UPDATE_EMA=0
        FREQ_STEP_UPDATE_EMA=1
        LAMBDA_GU=100.0
        python script_train.py \
            --version=v2 --dataset=single-mnist --dataset_category=8 \
            --ncs 128 64 32 8 --hidden_ncs 128 64 32 16 --middle_ns_flows 3 3 3 \
            --disc=vanilla --ndf=64 --nz=16 \
            --trigger_disc=5 --trigger_gen=1 \
            --trigger_gunif="${TRIG_GU}" --freq_step_gunif="${FREQ_STEP_GU}" --sn_power="${SN_POWER}" --lambda_gen_gunif="${LAMBDA_GU}" \
            --trigger_update_sv_ema="${TRIG_UPDATE_EMA}" --freq_step_update_sv_ema="${FREQ_STEP_UPDATE_EMA}" \
            --steps=200000 --batch_size=64 \
            --desc="single-mnist@8/gu@${TRIG_GU}#${FREQ_STEP_GU}#${SN_POWER}#${LAMBDA_GU}@${TRIG_UPDATE_EMA}#${FREQ_STEP_UPDATE_EMA}"
    fi
fi
