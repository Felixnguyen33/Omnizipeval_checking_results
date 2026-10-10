<h1 align="center"> WorfBench </h1>
<h3 align="center"> Benchmarking Agentic Workflow Generation </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2410.07869" target="_blank">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2410.07869" target="_blank">🤗HFPaper</a> •
  <a href="https://zjunlp.github.io/project/WorFBench/" target="_blank">🌐Web</a> •
  <a href="https://huggingface.co/collections/zjunlp/worfbench-66fc28b8ac1c8e2672192ea1" target="_blank">📊Dataset</a> •
  <a href="https://notebooklm.google.com/notebook/a4c13fd7-29da-462c-a47e-69a26c0d326e/audio" target="_blank">🎧NotebookLM Audio</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/WorFBench) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/WorFBench?color=green) 

## Table of Contents

- 🌻[Acknowledgement](#acknowledgement)
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- ✏️[Model-Inference](#model-inference)
- 📝[Workflow-Generation](#workflow-generation)
- 🤔[Workflow-Evaluation](#workflow-evaluation)
- 🚩[Citation](#citation)
<!-- - 🎉[Contributors](#🎉contributors) -->

---

## 🌻Acknowledgement

Our code of training module is referenced and adapted from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). And the Dataset is collected from [ToolBench](https://github.com/openbmb/toolbench?tab=readme-ov-file), [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca), [Lumos](https://github.com/allenai/lumos?tab=readme-ov-file), [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset), [Seal-Tools](https://github.com/fairyshine/seal-tools), [Alfworld](https://github.com/alfworld/alfworld), [Webshop](https://github.com/princeton-nlp/WebShop), [IntercodeSql](https://github.com/princeton-nlp/intercode). Our end-to-end evaluation module is based on [IPR](https://github.com/WeiminXiong/IPR), [Stable ToolBench](https://github.com/THUNLP-MT/StableToolBench). Thanks for their great contributions!



## 🌟Overview

Large Language Models (LLMs), with their exceptional ability to handle a wide range of tasks, have driven significant advancements in tackling reasoning and planning tasks, wherein decomposing complex problems into executable workflows is a crucial step in this process. Existing workflow evaluation frameworks either focus solely on holistic performance or suffer from limitations such as restricted scenario coverage, simplistic workflow structures, and lax evaluation standards. To this end, we introduce WorFBench, a unified workflow generation benchmark with multi-faceted scenarios and intricate graph workflow structures. Additionally, we present WorFEval, a systemic evaluation protocol utilizing subsequence and subgraph matching algorithms to accurately quantify the LLM agent's workflow generation capabilities. Through comprehensive evaluations across different types of LLMs, we discover distinct gaps between the sequence planning capabilities and graph planning capabilities of LLM agents, with even GPT-4 exhibiting a gap of around 15%. We also train two open-source models and evaluate their generalization abilities on held-out tasks. Furthermore, we observe that the generated workflows can enhance downstream tasks, enabling them to achieve superior performance with less time during inference. You can download our dataset from [huggingface](https://huggingface.co/collections/zjunlp/worfbench-66fc28b8ac1c8e2672192ea1)!

![](./assets/main_results.jpg)


## 🔧Installation

```bash
git clone https://github.com/zjunlp/WorFBench
cd WorFBench
pip install -r requirements.txt
```



## ✏️Model-Inference

We use [llama-facotry](https://github.com/hiyouga/LLaMA-Factory) to deploy local model with OpenAI-style API
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
```




## 📝Workflow-Generation
Generate workflow with local llm api
```bash
tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)
model_name=your_model_name
for task in ${tasks[@]}; do
    python node_eval.py \
        --task gen_workflow \
        --model_name ${model_name} \
        --gold_path ./gold_traj/${task}/graph_eval.json \
        --pred_path ./pred_traj/${task}/${model_name}/graph_eval_two_shot.json\
        --task_type ${task} \
        --few_shot \

done
```



## 🤔Workflow-Evaluation

Evaluation the workflow in the mode of *node* or *graph*
```bash
tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)
model_name=your_model_name
for task in ${tasks[@]}; do
    python node_eval.py \
        --task eval_workflow \
        --model_name ${model_name} \
        --gold_path ./gold_traj/${task}/graph_eval.json \
        --pred_path ./pred_traj/${task}/${model_name}/graph_eval_two_shot.json\
        --eval_model all-mpnet-base-v2 \
        --eval_output ./eval_result/${model_name}_${task}_graph_eval_two_shot.json \
        --eval_type node \
        --task_type ${task} \

done
```



## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{qiao2024benchmarking,
  title={Benchmarking Agentic Workflow Generation},
  author={Qiao, Shuofei and Fang, Runnan and Qiu, Zhisong and Wang, Xiaobin and Zhang, Ningyu and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2410.07869},
  year={2024}
}
```



<!-- ## 🎉Contributors

<a href="https://github.com/zjunlp/WorFBench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/WorFBench" /></a>

We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us. -->
