tasks=(os)
model_name=llama3
for task in ${tasks[@]}; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python node_eval.py \
        --task eval_workflow \
        --model_name ${model_name} \
        --gold_path ./gold_traj/${task}/graph_eval.json \
        --pred_path ./pred_traj/${task}/${model_name}/graph_eval_two_shot.json\
        --eval_model all-mpnet-base-v2 \
        --eval_output ./eval_result/${model_name}_${task}_graph_eval_two_shot.json \
        --eval_type node \
        --task_type ${task} \
        --few_shot \

done
