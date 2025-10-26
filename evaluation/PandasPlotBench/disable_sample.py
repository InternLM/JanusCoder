import json

source_fp = '/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/PandasPlotBench/out_results/results_qwen3_4b_base_mclm_exp10_matplotlib_head_1.json'
dump_fp = '/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/PandasPlotBench/out_results/results_qwen3_4b_base_mclm_exp10_matplotlib_head_2.json'
disable_sample = '76'
with open(source_fp, 'r') as f_in:
    with open(dump_fp, 'w') as f_out:
        data = json.load(f_in)
        data['code'][disable_sample] = "'''\n" + data['code'][disable_sample] + "\n'''"
        json.dump(data, f_out, ensure_ascii=False)

