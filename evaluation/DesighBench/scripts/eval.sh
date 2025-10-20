source /cpfs01/shared/XNLP_H800/miniconda3/bin/activate /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/conda-envs/designbench
cd /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench
python /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/code/eval_vlm.py \
    --model_name "final0_internvl3.5-8b_0820-vistune" \
    --base_url "http://10.130.128.31:xxxx/v1" \
    --api_key "sk-xxxxxxxxxxxxxx" \
    --is_lite 0 \
    --use_runner 1 \
    --mode "both" 