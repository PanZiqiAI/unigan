
if [ "$1" == "baseline" ]; then
    python script_train.py \
        --dataset=celeba-hq \
        --ncs 512 256 128 64 16 --hidden_ncs 512 256 128 64 32 --middle_ns_flows 3 3 3 3 \
        --disc=stylegan2 --nz=64 \
        --trigger_disc=5 --trigger_gen=1 \
        --freq_step_gunif=-1 \
        --freq_step_update_sv_ema=-1 \
        --steps=400000 --batch_size=96 \
        --freq_counts_step_eval_vis=1000 --freq_counts_step_eval_jacob=10 --freq_counts_step_chkpt=50 \
        --desc="celeba-hq/baseline"
elif [ "$1" == "gunif" ]; then
    TRIG_GU=0
    FREQ_STEP_GU=10
    SN_POWER=3
    LAMBDA_GU=1.0
    TRIG_UPDATE_EMA=0
    FREQ_STEP_UPDATE_EMA=50
    python script_train.py \
        --version=v2 --dataset=celeba-hq \
        --ncs 512 256 128 64 16 --hidden_ncs 512 256 128 64 32 --middle_ns_flows 3 3 3 3 \
        --disc=stylegan2 --nz=64 \
        --trigger_disc=5 --trigger_gen=1 \
        --trigger_gunif="${TRIG_GU}" --freq_step_gunif="${FREQ_STEP_GU}" --sn_power="${SN_POWER}" --lambda_gen_gunif="${LAMBDA_GU}" \
        --trigger_update_sv_ema="${TRIG_UPDATE_EMA}" --freq_step_update_sv_ema="${FREQ_STEP_UPDATE_EMA}" \
        --steps=400000 --batch_size=96 \
        --freq_counts_step_eval_vis=1000 --freq_counts_step_eval_jacob=10 --freq_counts_step_chkpt=50 \
        --desc="celeba-hq/gu@${TRIG_GU}#${FREQ_STEP_GU}#${SN_POWER}#${LAMBDA_GU}@${TRIG_UPDATE_EMA}#${FREQ_STEP_UPDATE_EMA}"
fi
