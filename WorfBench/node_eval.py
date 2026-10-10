import json
from LLM.localLLM import localLLM
from typing import List
import argparse
import os
import re
from tqdm import tqdm
from evaluator.graph_evaluator import t_eval_graph,t_eval_nodes
from prompts.eval_prompt import two_shot_example as example

# OmniZipEval: node-line regex as a module-level setting. The default is the
# upstream regex (a space is required after "1:" / "1."), so upstream behaviour
# is unchanged; scripts/omnizipeval/generate.py sets it to r'\d+[:.]\s*(.+)'.
NODE_PATTERN = re.compile(r'\d+[:.] (.+)')


def workflow_to_node_list(workflow: str) -> List[str]:
    if "Node" not in workflow:
        print("workflow is not in the right format")
        return []
    matches = NODE_PATTERN.findall(workflow)

    workflow = [match.strip() for match in matches]
    if len(workflow) != 0 and ("Finish" in workflow[-1] or "finish" in workflow[-1]):
        workflow = workflow[:-1]
    elif len(workflow) == 0:
        print("workflow is empty")
    return workflow

def workflow_to_graph_list(workflow: str) -> List[str]:
    try:
        if "Node" not in workflow:
            print("workflow is not in the right format")
            return []
        
        node_matches = NODE_PATTERN.findall(workflow)

        node_workflow = [match.strip() for match in node_matches]
        if len(node_workflow) != 0 and ("Finish" in node_workflow[-1] or "finish" in node_workflow[-1]):
            # node_workflow = node_workflow[:-1]
            pass
        elif len(node_workflow) == 0:
            print("node_workflow is empty")

        node_workflow.insert(0,"START")
        node_workflow.append("END")
        
        edge_pattern = re.compile(r'\(\s*(\d+|START)\s*,\s*(\d+|END)\s*\)')

        edge_matches = edge_pattern.findall(workflow)

        if len(edge_matches) == 0:
            print("edge_workflow is empty")
            return []
        
        edge_workflow = []
        for i, match in enumerate(edge_matches):
            edge = list(match)
            if "START" in edge:
                edge[edge.index("START")] = "0"
            if "END" in edge:
                edge_num = len(node_workflow) - 1
                edge[edge.index("END")] = str(edge_num)
            edge = tuple(map(int, edge))  # Convert back to tuple after modification
            edge_workflow.append(edge)
            
        # print(f"edge_workflow: {edge_workflow}")
        if len(edge_workflow) == 0:
            print("edge_workflow is empty")
            return []
        
        workflow = {"nodes":node_workflow,"edges":edge_workflow}
        return workflow
    except Exception as e:
        print(e)
        return []


def build_message(query:str, few_shot:bool, task_type:str):
    messages = []
    messages.append(query[0])
    if few_shot:
        messages.extend(example[task_type])
    
    for i in range(1,len(query)-1):
        if i % 2 == 1:
            query[i]["content"] = "Now it's your turn.\n" + query[i]["content"]
            messages.append(query[i])
        else:
            messages.append(query[i])
    return messages


def gen_workflow(gold_path:str,pred_path:str,model_name:str,few_shot:bool,task_type:str):
    with open(gold_path,"r") as f:
        eval_data = json.load(f)
    if os.path.exists(pred_path):
        with open(pred_path, "r") as pred_file:
            preded_datas = json.load(pred_file)
            start = len(preded_datas)
            eval_data = eval_data[start:]
            workflow_list = preded_datas
    else:
        pred_dir = "/".join(pred_path.split("/")[:-1])
       
        if not os.path.exists(pred_dir):
            
            os.mkdir(pred_dir)

        workflow_list = []
    workflow_list = []
    with tqdm(total=len(eval_data)) as pbar:
        for data in eval_data:
            messages = build_message(data['conversations'],few_shot,task_type)
            # print(messages)
            plan = localLLM(messages=messages,api_port=8000)
            workflow_list.append({'query':data,'workflow':plan})
            pbar.update(1)      
            print(plan)
            print("===========================")  
    with open(pred_path,"w") as f:
        json.dump(workflow_list,f,indent=4,ensure_ascii=False)
        print("save workflow_list.json successfully")





def eval_workflow(gold_path:str,pred_path:str,eval_model:str, eval_type:str, eval_output:str):
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer(eval_model)

    eval_output_dir = "/".join(eval_output.split("/")[:-1])
    if not os.path.exists(eval_output_dir):
        os.mkdir(eval_output_dir)

    with open(gold_path,"r") as f:
        gold_data = json.load(f)
    # end with json
    if pred_path.endswith(".json"):
        with open(pred_path,"r") as f:
            pred_data = json.load(f)
            
    #end with jsonl
    elif pred_path.endswith(".jsonl"):
        pred_data = []
        with open(pred_path,"r") as f:
            for line in f:
                pred_data.append(json.loads(line))
    gold_plan = [d['conversations'][-1]["content"] for d in gold_data]
    if pred_path.endswith(".json"):
        pred_plan = [d["workflow"] for d in pred_data]
    elif pred_path.endswith(".jsonl"):
        pred_plan = [d["answer"]["choices"][0]["message"]["content"] for d in pred_data]
    # print(f"gold_plan:{len(gold_plan)}")
    # print(f"pred_plan:{len(pred_plan)}")
    assert len(gold_plan) == len(pred_plan) , "The number of gold plan and pred plan should be the same"
    #caluate average precision,recall,f1_score
    all_precision = 0
    all_recall = 0
    all_f1_score = 0
    with tqdm(total=len(gold_plan)) as pbar:
        for i in range(len(gold_plan)):
            eval_result = {}
            if eval_type == "node":
                gold_graph_workflow = workflow_to_graph_list(gold_plan[i])
                pred_graph_workflow = workflow_to_graph_list(pred_plan[i])
                # print(f"gold_graph_workflow:{gold_graph_workflow}")
                # print(f"pred_graph_workflow:{pred_graph_workflow}")
                if pred_graph_workflow == []:
                    continue
                eval_result = t_eval_nodes(pred_graph_workflow, gold_graph_workflow, sentence_model)
            elif eval_type == "graph":
                gold_graph_workflow = workflow_to_graph_list(gold_plan[i])
                pred_graph_workflow = workflow_to_graph_list(pred_plan[i])
                # print(f"gold_graph_workflow:{gold_graph_workflow}")
                # print(f"pred_graph_workflow:{pred_graph_workflow}")
                if pred_graph_workflow == []:
                    continue
                eval_result = t_eval_graph(pred_graph_workflow, gold_graph_workflow, sentence_model)
            
            all_precision += eval_result['precision']
            all_recall += eval_result['recall']
            all_f1_score += eval_result['f1_score']
            pbar.update(1)
    all_precision /= len(gold_plan)
    all_recall /= len(gold_plan)
    all_f1_score /= len(gold_plan)
    result = {"precision":all_precision,"recall":all_recall,"f1_score":all_f1_score}
    json.dump(result, open(eval_output,"w"),indent=4,ensure_ascii=False)
    print(f"Average Precision:{all_precision}")
    print(f"Average Recall:{all_recall}")
    print(f"Average F1_score:{all_f1_score}")
    print("=========================================")


def main(args):
    print(args.task_type)
    if args.task == "gen_workflow":
        gen_workflow(args.gold_path,args.pred_path,args.model_name,args.few_shot,args.task_type)
    elif args.task == "eval_workflow":
        eval_workflow(args.gold_path,args.pred_path,args.eval_model,args.eval_type,args.eval_output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["gen_workflow", "eval_workflow"], default="gen_workflow")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task_type", type=str, default="wikihow")
    parser.add_argument("--gold_path", type=str, default="./gold_traj/wikihow/graph_eval.json")
    parser.add_argument("--pred_path", type=str, default="./pred_traj/wikihow/llama3/graph_eval_two_shot.json")
    parser.add_argument("--eval_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--eval_output", type=str, default="./eval_result/node/llama3/wikihow_node_eval_two_shot.json")
    parser.add_argument("--eval_type", choices=["node","graph"], default="node")
    parser.add_argument("--few_shot", action='store_true')
    args = parser.parse_args()
    main(args)
