echo "!!! FINAL RUN !!!"

id="final_camera_ready_run"

mkdir -p ./${id}/
mkdir -p experiments/benchmarks/

python3 run_experiments.py fff --max_time 5 --episodes 100 --id ${id}_fff_5s --multi 34 | tee ${id}/${id}_fff_5s.log
python3 run_experiments.py mars --max_time 15 --episodes 100 --id ${id}_mars_exp_15s --multi 34 | tee ${id}/${id}_mars_exp_15s.log
python3 run_experiments.py ct --max_time 5 --episodes 100 --id ${id}_ct_exp_5s --multi 34 | tee ${id}/${id}_ct_exp_5s.log
