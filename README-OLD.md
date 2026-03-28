<div align="center">

# ***ReSearch***: Learning to ***Re***ason with ***Search*** for LLMs via Reinforcement Learning

[![Arxiv](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2503.19470) [![Model](https://img.shields.io/badge/model-4169E1?style=for-the-badge&logo=huggingface)](https://huggingface.co/collections/agentrl/research-67e506a0311bea06dc54878b) 

</div>

<p align="center">
<img src="./assets/intro_bar.png" width="90%" alt="Intro" />
<img src="./assets/method.png" width="90%" alt="Method" />
</p>

We propose ***ReSearch***, a novel framework that trains LLMs to ***Re***ason with ***Search*** via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning.

## 📰 News
- **[2025-03-27]** 🤗 We release our trained models on [Hugging Face](https://huggingface.co/collections/agentrl/research-67e506a0311bea06dc54878b), please check it out! 
- **[2025-03-26]** 🎉 We release the paper, update the code and open-source the models.
  - 📝 The **paper is released** on arXiv, more details and evaluation results can be found in our [paper](https://arxiv.org/abs/2503.19470).
  - 🛠️ The **repository is updated** with the new implementation, especially the rollout with search during RL training. This version of implementation is based on the latest release of verl.
- **[2025-03-03]** ✅ We have released the preview version of ReSearch implementation.

## 📦 Installation

We recommend using conda to manage the environment. First create a conda environment and activate it.
```bash
conda create -n re-search python==3.10
conda activate re-search
```
Then install dependencies, and our modified verl and flashrag packages  under ```src/``` will be installed in the editable mode.  Check out ```setup.py``` for details.
```bash
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
git clone https://github.com/Agent-RL/ReSearch.git
cd ReSearch
pip3 install -e .
```
As described in the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#wrench-installation), due to the incompatibility when installing faiss using pip, we need to use the following conda command to install faiss-gpu.
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## 🚀 Quick Start

### Retriever Serving

As described in our paper, during model training and evaluation, search operation will be conducted in the rollout and inference process. In practice, we host a retriever service via FlashRAG and FastAPI. Hence, the search operation is standardized to be an API call. This serving can be used to decouple the search operation from the reinforcement learning process, making the training and evaluation more clear and flexible.

Before starting the retriever serving, you need download the [pre-indexed wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [wikipedia corpus and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). More details can be found in the documentation of FlashRAG.

For starting the retriever serving, you need to first fill the `scripts/serving/retriever_config.yaml` with the correct path to the retrieval model, index, and corpus, and available GPU ids. Then, you can run the following command to start the retriever serving:
```bash
cd scripts/serving
python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever {num_retriever} \  
    --port {port}
```

The started retriever serving will be used in the training and evaluation process in the following part.

### Data Preparation

*ReSearch* is trained on the training set of MuSiQue, and evaluated on the dev set of HotpotQA, 2WikiMultiHopQA, MuSiQue and Bamboogle. For downloading the datasets, please refer to the `data/download_dataset.sh` script.
```bash
cd data
bash download_dataset.sh
```

For preparing the training and validation data for following reinforcement learning, please run this script to parse the MuSiQue dataset to the parquet format.
```bash
cd data
python prepare_musique.py
```

### Training

Our training framework is based on [verl](https://github.com/volcengine/verl), a powerful reinforcement learning framework for LLMs. We deeply customize the verl code to fit our needs, and the modified version of verl is under the `src/verl` directory. The example of training scripts are under `scripts/train`.

#### Single-node training
Here is an example of training Qwen2.5-7B-Instruct with 4 GPUs locally. Note that the training script below **is just an example** for single-node training, using small batch size for quick start, and do not assure the training performance.
```bash
cd scripts/train
bash train.sh \
    --train_batch_size 8 \
    --ppo_mini_batch_size 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {model/path/to/qwen2.5-7b-instruct} \
    --search_url {your-hosted-retriever-url} \
    --project_name {wandb-project-name} \
    --experiment_name {wandb-experiment-name} \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 5 \
    --test_freq 5 \
    --total_epochs 2 \
    --wandb_api_key {your-wandb-api-key} \
    --save_path {path/to/save} \
    --train_files {path/to/train/parquet/data} \
    --test_files {path/to/test/parquet/data}
```
- For training base (pre-trained) models, please use `--apply_chat False` and `--prompt_template_name re_search_template`
- For training instruction-tuned models, please use `--apply_chat True` and `--prompt_template_name re_search_template_sys`

#### Multi-node training

If you want to **fully reproduce** the results in our paper, please refer to the multi-node training script in `scripts/train/train_multi_node.sh`, as well as the implementation details in our paper.

### Evaluation

We recommend using [SGLang](https://docs.sglang.ai/) to serve the trained model. You can download our open-sourced models or trained your own models to conduct the evaluation. Here is an example of launching the model serving:
```bash
python3 -m sglang.launch_server \
        --served-model-name {trained/model/name} \
        --model-path {trained/model/path} \
        --tp 2 \
        --context-length 8192 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 80 \
        --trust-remote-code \
        --disable-overlap \
        --disable-radix-cache
```

We use [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) as the standard evaluation environment. Here is an example of evaluating the performance of ReSearch-Qwen-7B-Instruct on Bamboogle test set.
```bash
cd scripts/evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --method_name research \
    --data_dir {root/path/to/evaluation/data} \
    --dataset_name bamboogle \
    --split test \
    --save_dir {your-save-dir} \
    --save_note research_qwen7b_ins
    --sgl_remote_url {your-launched-sgl-url} \
    --remote_retriever_url {your-hosted-retriever-url} \
    --generator_model {your-local-model-path} \
    --apply_chat True
```

For base model, please use `--apply_chat False` and for instruction-tuned model, please use `--apply_chat True`, for loading correct prompt template when conducting evaluation for *ReSearch* model. For more details about the configuration, please refer to the `scripts/evaluation/eval_config.yaml` file. 

## 🤝 Acknowledge

This training implementation is based on [verl](https://github.com/volcengine/verl) and the evaluation is based on [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). The serving of retriever is based on [FastAPI](https://github.com/fastapi/fastapi). The model serving is based on [SGLang](https://docs.sglang.ai/). *ReSearch* models are trained based on [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/). We sincerely appreciate their contributions to the open-source community.

## 📚 Citation

If you find this work useful, please cite it as follows:
```bibtex
@misc{chen2025research
  title={ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning}, 
  author={Mingyang Chen and Tianpeng Li and Haoze Sun and Yijie Zhou and Chenzheng Zhu and Haofen Wang and Jeff Z. Pan and Wen Zhang and Huajun Chen and Fan Yang and Zenan Zhou and Weipeng Chen},
  year={2025},
  eprint={2503.19470},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2503.19470}, 
}
```