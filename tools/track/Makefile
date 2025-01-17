# Path to the original V2X-Sim dataset
original_data_path := /data/v2x-sim
# Path to the dataset for detection
det_data_path := /data/v2x-sim-nyu/V2X-Sim-det
# [lowerbound / upperbound/ v2v / disco / when2com / when2com_warp / who2com / who2com_warp]
mode := when2com
# [with_rsu / no_rsu]
rsu := no_rsu
# Index of current agent
current_agent := 0
# [train / test / val]
split := test
# det path
det_path := ~/code/coperception_mod/tools/det
# file containing idxes of scenes to run tracking
scene_idxes_file := ~/code/coperception_mod/tools/utils/$(split)_scenes.txt
# Index of the start agent
from_agent := 1
# Index of the end agent + 1
to_agent := 6
# Det logs path (to get the tracking input)
det_check_path:= check_loss
det_logs_path := $(det_path)/${det_check_path}
# start of bootstrap for testing
boot_start := 0
# end of bootstrap for testing, include the boot_end 25
boot_end := 25
output_name := name
nll_threshold := 1000
max_age := 1

.PHONY: sort

create_data:
	python create_data_com.py --root $(original_data_path) --data $(det_data_path)/$(split) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file)

sort:
	cd sort && python sort.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --max_age $(max_age)
    
mbb_sort:
	cd sort && python sort_mbb.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --bootstrap_start $(boot_start) --bootstrap_end ${boot_end}

eval:
	python run_multi_agent_mot_challenge.py --mode $(mode) --rsu $(rsu) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --split $(split) --name $(output_name)

mbb_eval:
	python mbb_run_multi_agent_mot_challenge.py --mode $(mode) --rsu $(rsu) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --split $(split) --bootstrap_start $(boot_start) --bootstrap_end ${boot_end}
    
sort_no:
	cd sort && python sort_uq.py --mode $(mode)/$(rsu) --split $(split)  --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --max_age $(max_age)
    
sort_cfk:
	cd sort && python sort_uq.py --mode $(mode)/$(rsu) --split $(split)  --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov --max_age $(max_age)

sort_nlla:
	cd sort && python sort_uq.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold) --max_age $(max_age)

sort_cfk_nlla:
	cd sort && python sort_uq.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov --nll_ass --nll_threshold $(nll_threshold) --max_age $(max_age)

sort_uq_output:
	cd sort && python sort_uq_out.py --mode $(mode)/$(rsu) --split $(split) --output_cov --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold) --max_age $(max_age)
    
sort_output:
	cd sort && python sort_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --max_age $(max_age)
    

byte_track:
	cd byte_tracker && python byte_main.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path)
    
byte_track_cfk:
	cd byte_tracker && python byte_main.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov
    
byte_track_nlla:
	cd byte_tracker && python byte_main.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold)
    
byter_track_cfk_nlla:
	cd byte_tracker && python byte_main.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold) --output_cov
    
byte_track_out:
	cd byte_tracker && python byte_main_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path)
    
byte_track_cfk_out:
	cd byte_tracker && python byte_main_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov
    
byte_track_nlla_out:
	cd byte_tracker && python byte_main_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold)
    
byter_track_cfk_nlla_out:
	cd byte_tracker && python byte_main_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold) --output_cov
    
sort_no_out:
	cd sort && python sort_uq_out.py --mode $(mode)/$(rsu) --split $(split)  --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --max_age $(max_age)
    
sort_cfk_out:
	cd sort && python sort_uq_out.py --mode $(mode)/$(rsu) --split $(split)  --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov --max_age $(max_age)

sort_nlla_out:
	cd sort && python sort_uq_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --nll_ass --nll_threshold $(nll_threshold) --max_age $(max_age)

sort_cfk_nlla_out:
	cd sort && python sort_uq_out.py --mode $(mode)/$(rsu) --split $(split) --from_agent $(from_agent) --to_agent $(to_agent) --scene_idxes_file $(scene_idxes_file) --det_logs_path $(det_logs_path) --output_cov --nll_ass --nll_threshold $(nll_threshold) --max_age $(max_age)