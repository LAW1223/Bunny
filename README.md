<h1 align = "center">
  Multimodal-Robustness-Benchmark
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.04264">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2406.04264-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-MMR Benchmark-yellow">
    </a>
    <a href="https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-yellow">
    </a>
</p>

This repo contains the official evaluation code and dataset for the paper“Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions”.

## 📢 News and Updates

* 2024.06.15 🔥 **ArXiv paper is released!**
* 2024.06.13 🔥 **MMR benchmark and MMR-data are released!**

## 📇 Contents
- [MMR-benchmark](#%EF%B8%8F-mmr-benchmark)
- [Evaluation](#-evaluation)
- [Leaderboard](#-leaderboard)
- [MMR-data](#-mmr-data)
- [Training](#-training)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgement](#-acknowledgement)

## ⚖️ MMR-benchmark

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in visual understanding and reasoning, providing reasonably accurate answers, such as image descriptions. This has spurred extensive research into evaluating MLLMs. Most evaluation benchmarks assume that incorrect answers indicate a lack of understanding of the visual content. However, our findings reveal that, in many cases, MLLMs answer questions incorrectly despite correctly understanding the visual content. This suggests that incorrect answers do not necessarily imply a lack of comprehension but may instead result from a lack of robustness to leading questions.

To comprehensively measure MLLMs' understanding capability and robustness to leading questions, we introduce a multi-modal robustness benchmark (MMR). MMR contains paired positive and negative questions across 12 categories, meticulously annotated by humans. We manually construct 300 positive and 300 leading negative questions across three levels: character, attribute, and context. Character-level questions prompt identifying elements like characters or numbers, while attribute-level questions focus on properties such as color, texture, and quantity. Context-level inquiries delve into higher-level concepts like emotions, culture, and common sense. The positive questions aim to evaluate the model's understanding ability, while the misleading ones challenge its resistance to interference.

<p align="center">
  <img src="./MMA_benchmark.png" alt="Logo">
</p>

## 🏁 Evaluation

Please refer to our [evaluation](https://github.com/FlagOpen/FlagEmbedding/tree/master/MLVU/evaluation) folder for more details.

## 🏆 Leaderboard

| Method                          | Char/Num | Pres.  | Color/Tex | Num.  | Shape | Posture | Pos.  | Abstract. | Concrete. | Expert. | Act.  | Rel. | Avg. RA ↑ |
|---------------------------------|----------|--------|-----------|-------|-------|---------|-------|-----------|-----------|---------|-------|------|------------|
| GPT-4o 🥇            | 72.50    | 68.18  | 66.67     | 45.83 | 87.5  | 70.83   | 50.00 | 68.18     | 76.19     | 70.97   | 83.33 | 63.64| 69.00      |
| Mini-Gemini-HD-34B 🥇 | 62.50    | 63.64  | 70.83     | 54.17 | 79.17 | 62.50   | 72.73 | 86.36     | 85.71     | 54.84   | 19.17 | 68.18| 69.00      |
| LLaVA-1.6-34B 🥉 | 75.00    | 68.18  | 66.67     | 41.67 | 79.17 | 54.17   | 72.72 | 81.81     | 71.42     | 64.52   | 79.17 | 68.18| 68.67      |
| Qwen-VL-max | 67.50    | 72.73  | 66.67     | 41.67 | 79.17 | 62.5    | 63.64 | 77.27     | 80.95     | 61.29   | 79.17 | 72.73| 68.33      |
| Bunny-Llama-3-8B-V | 55.00    | 63.64  | 54.17     | 37.50 | 79.17 | 62.50   | 54.55 | 72.73     | 85.71     | 48.39   | 75.00 | 50.00| 60.67      |
| Step-1V          | 60.00    | 54.55  | 58.33     | 20.83 | 70.83 | 54.17   | 31.82 | 54.55     | 57.14     | 45.16   | 79.17 | 50.00| 53.33      |
| **Method**                      | Char/Num | Pres.  | Color/Tex | Num.  | Shape | Posture | Pos.  | Abstract. | Concrete. | Expert. | Act.  | Rel. | Avg. MR ↓ |
| Mini-Gemini-HD-34B 🥇 | 20.3     | 20.3   | 21.0      | 22.1  | 14.8  | 19.9    | 18.4  | 15.3      | 16.0      | 19.7    | 28.3  | 19.1 | 19.3       |
| LLaVA-1.6-34B 🥈 | 18.5     | 20.4   | 24.7      | 25.0  | 16.6  | 23.2    | 21.3  | 16.2      | 18.7      | 21.2    | 19.5  | 19.4 | 20.1       |
| GPT-4o  🥉         | 21.6     | 21.7   | 24.8      | 24.0  | 14.7  | 19.7    | 22.2  | 21.3      | 19.6      | 18.5    | 18.6  | 20.3 | 20.6       |
| Bunny-Llama-3-8B-V | 23.5     | 21.1   | 23.5      | 23.2  | 17.0  | 19.7    | 24.4  | 19.2      | 16.0      | 25.3    | 19.6  | 24.3 | 21.2       |
| Qwen-VL-max | 21.9     | 21.1   | 24.7      | 26.5  | 16.0  | 22.8    | 24.0  | 21.3      | 19.2      | 23.0    | 20.4  | 19.7 | 21.5       |
| Step-1V           | 24.8     | 24.1   | 26.7      | 32.7  | 19.8  | 24.5    | 29.3  | 23.2      | 23.0      | 29.6    | 18.4  | 26.8 | 25.2       |

## 🚩 MMR-data

To enhance MLLMs' understanding capability and robustness, we propose a data construction method using GPT-4V to generate paired positive and negative samples for instruction tuning. The method includes three steps: 1) Information extraction. We implicitly and comprehensively extract detailed information from images, including text, object attributes, human characteristics, relationships between objects, relationships between people, events, and overall perception. 2) Instruction tuning data generation. We generate positive samples using the extracted information and construct negative samples that directly contradict the positive ones. 3) Sample filtering. We filter samples through keyword matching to remove those with uncertain answers and redundant phrases.

<p align="center">
  <img src="./data_collection.png" alt="Logo">
</p>

## 🤖 Training

- We build the model based on [Bunny](https://github.com/BAAI-DCAI/Bunny). Please refer to [Bunny](https://github.com/BAAI-DCAI/Bunny) for more details.
- Training details and checkpoints.
  
| Checkpoint                                                   | Vision Encoder                                               | LLM                                                          | Pretrain lr | Pretrain weights                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :---------: | ------------------------------------------------------------ |
| Bunny-MMR-3B | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)    |    5e-4     | [bunny-pretrain-phi-2-siglip](https://huggingface.co/BAAI/bunny-pretrain-phi-2-siglip) |
| Bunny-MMR-4B | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |    1e-3     | [bunny-pretrain-phi-3-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-phi-3-siglip) |
| Bunny-MMR-8B | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |    1e-3     | [bunny-pretrain-llama3-8b-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-llama3-8b-siglip) |

## 🔗 Citation
If you find this repository helpful, please cite the paper below.

```bibtex
@article{he2024bunny,
      title={Efficient Multimodal Learning from Data-centric Perspective}, 
      author={He, Muyang and Liu, Yexin and Wu, Boya and Yuan, Jianhao and Wang, Yueze and Huang, Tiejun and Zhao, Bo},
      journal={arXiv preprint arXiv:2402.11530},
      year={2024}
}
```

## 🧾 License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).

## 📫 Acknowledgement

We build our project based on [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant.
