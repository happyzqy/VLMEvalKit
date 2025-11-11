cd /disk/zdata1/home/zhangqingyu/work_guo/gui/VLMEvalKit

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# python run.py --data ScreenSpot --model Qwen3-VL-2B-Instruct --verbose
python run.py --data AITZ_LOCAL --model Qwen3-VL-2B-Instruct --verbose


#torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
