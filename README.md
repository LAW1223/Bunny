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
| GPT-4o 🥇                       | 72.50    | 68.18  | 66.67     | 45.83 | 87.5  | 70.83   | 50.00 | 68.18     | 76.19     | 70.97   | 83.33 | 63.64| 69.00      |
| Mini-Gemini-HD-34B 🥇            | 62.50    | 63.64  | 70.83     | 54.17 | 79.17 | 62.50   | 72.73 | 86.36     | 85.71     | 54.84   | 19.17 | 68.18| 69.00      |
| LLaVA-1.6-34B 🥉                 | 75.00    | 68.18  | 66.67     | 41.67 | 79.17 | 54.17   | 72.72 | 81.81     | 71.42     | 64.52   | 79.17 | 68.18| 68.67      |
| Qwen-VL-max                       | 67.50    | 72.73  | 66.67    | 41.67  | 79.17 | 62.5 | 63.64  | 77.27      | 80.95     | 61.29   | 79.17  | 72.73 | 68.33     |
| Bunny-Llama-3-8B-V               | 55.00    | 63.64  | 54.17     | 37.50 | 79.17 | 62.50   | 54.55 | 72.73     | 85.71     | 48.39   | 75.00 | 50.00| 60.67      |
| InternVL-Chat-V1-5 (26B)         | 62.5     | 59.09  | 66.67     | 41.67 | 66.67 | 41.67   | 54.55 | 63.64     | 66.67     | 45.16   | 79.17 | 72.73| 59.67      |
| Yi-VL-34B                        | 52.50    | 63.64  | 70.83     | 41.67 | 75.00 | 37.50   | 59.09 | 68.18     | 57.14     | 48.39   | 70.83 | 63.64| 58.33      |
| Bunny-MMR-3B                     | 60.0     | 59.09  | 58.33     | 25.0  | 83.33 | 50.0    | 54.55 | 68.18     | 57.14     | 51.61   | 79.17 | 54.55| 58.33      |
| Idefics2-8B                      | 57.50    | 59.09  | 54.17     | 50.00 | 79.17 | 41.67   | 27.27 | 77.27     | 76.19     | 45.16   | 75.00 | 40.91| 56.67      |
| Cogvlm2-llama3                   | 60.00    | 63.64  | 54.17     | 37.5  | 70.83 | 33.33   | 40.91 | 50.00     | 85.71     | 41.94   | 62.50 | 50.00| 54.00      |
| Step-1V                          | 60.00    | 54.55  | 58.33     | 20.83 | 70.83 | 54.17   | 31.82 | 54.55     | 57.14     | 45.16   | 79.17 | 50.00| 53.33      |
| Phi-3-vision (4B)                | 62.50    | 59.09  | 58.33     | 37.50 | 70.83 | 33.33   | 31.82 | 54.55     | 66.67     | 41.94   | 58.33 | 50.00| 52.33      |
| Glm-4V                           | 60.00    | 54.55  | 54.17     | 29.17 | 58.33 | 41.67   | 27.27 | 72.73     | 47.62     | 35.48   | 70.83 | 45.45| 50.00      |
| Gemini-pro-vision                | 42.50    | 50.00  | 41.67     | 25.00 | 83.33 | 50.00   | 45.45 | 40.91     | 47.62     | 45.16   | 70.83 | 45.45| 48.67      |
| Deepseek-VL-7B-Chat              | 52.50    | 54.55  | 54.17     | 37.5  | 62.5  | 25.00   | 18.18 | 54.55     | 52.38     | 35.48   | 75.00 | 50.00| 47.67      |
| Mplug-owl2-llama2-7B             | 32.50    | 63.64  | 58.33     | 20.83 | 62.50 | 37.50   | 13.64 | 54.55     | 47.62     | 25.81   | 58.33 | 31.82| 41.33      |
| MiniCPM-Llama3-V                 | 37.5     | 45.45  | 50.00     | 16.67 | 41.67 | 37.5    | 36.36 | 68.18     | 33.33     | 29.03   | 41.67 | 54.55| 40.33      |
| LLaVA-RLHF (7B)                  | 7.50     | 36.36  | 33.33     | 33.33 | 50.00 | 16.67   | 9.09  | 59.09     | 38.10     | 22.58   | 50.00 | 31.82| 30.67      |
| Claude3-Opus-V                   | 35.00    | 22.73  | 12.50     | 16.67 | 33.33  | 16.67  | 22.73 | 45.45     | 33.33     | 25.81   | 37.50 | 40.91| 28.67      |


| Method                     | Char/Num | Pres.  | Color/Tex | Num.  | Shape | Posture | Pos.  | Abstract. | Concrete. | Expert. | Act.  | Rel. | Avg. MR ↓ |
|---------------------------------|----------|--------|-----------|-------|-------|---------|-------|-----------|-----------|---------|-------|------|------------|
| Mini-Gemini-HD-34B              | 21.88    | 12.50  | 10.53     | 7.14  | 5.00  | 28.57   | 15.79 | 9.52      | 5.26       | 32.00   | 9.52  | 11.76 | 15.16                |
| LLaVA-1.6-34B                   | 6.25     | 11.76  | 20.00     | 23.08 | 9.52  | 35.00   | 11.11 | 14.28     | 25.00      | 20.00   | 9.52  | 16.67 | 16.26                |
| GPT-4o                          | 9.38     | 16.67  | 23.81     | 26.67 | 4.55  | 19.05   | 38.89 | 28.57     | 15.79      | 24.14   | 13.04 | 22.22 | 19.46                |
| Qwen-VL-max                     | 22.86    | 11.11  | 23.81     | 28.57 | 5.00  | 25.00   | 30.00 | 19.05     | 19.05      | 29.63   | 9.52  | 15.79 | 20.23                |
| Bunny-Llama-3-8B-V              | 15.38    | 22.22  | 18.75     | 40.00 | 5.00  | 28.57   | 29.41 | 23.81     | 10.00      | 40.00   | 10.00 | 26.67 | 22.22                |
| Bunny-MMR-3B                    | 11.11    | 13.33  | 26.32     | 53.85 | 4.76  | 40.00   | 29.41 | 28.57     | 33.33      | 33.33   | 9.52  | 14.29 | 23.91                |
| Idefics2-8B                     | 23.33    | 27.78  | 23.53     | 20.00 | 13.64 | 50.00   | 40.00 | 22.73     | 11.11      | 41.67   | 14.29 | 40.00 | 26.72                |
| Yi-VL-34B                       | 27.59    | 22.22  | 15.00     | 28.57 | 10.00 | 50.00   | 27.78 | 16.67     | 42.86      | 42.31   | 22.73 | 17.65 | 27.39                |
| InternVL-Chat-V1-5 (26B)        | 21.88    | 18.75  | 23.81     | 37.50 | 27.27 | 52.38   | 29.41 | 33.33     | 26.32      | 44.00   | 13.64 | 20.00 | 28.97                |
| Step-1V                         | 14.29    | 25.00  | 26.32     | 61.54 | 5.56  | 40.91   | 61.11 | 33.33     | 33.33      | 44.00   | 9.52  | 21.43 | 30.43                |
| Cogvlm2-llama3                  | 22.58    | 22.22  | 27.78     | 30.77 | 15.00 | 61.90   | 35.71 | 45.00     | 14.29      | 51.85   | 28.57 | 38.89 | 33.06                |
| Phi-3-vision (4B)               | 19.35    | 18.75  | 26.32     | 43.75 | 19.05 | 55.56   | 58.82 | 40.00     | 22.22      | 48.00   | 33.33 | 31.25 | 34.03                |
| Gemini-pro-vision               | 29.17    | 31.25  | 41.18     | 45.45 | 13.04 | 40.00   | 33.33 | 52.63     | 44.44      | 48.15   | 19.05 | 23.08 | 34.82                |
| Glm-4V                          | 27.27    | 36.84  | 35.00     | 56.25 | 33.33 | 52.38   | 53.85 | 20.00     | 47.37      | 57.69   | 19.05 | 37.50 | 38.78                |
| Deepseek-VL-7B-Chat             | 30.00    | 20.00  | 27.78     | 43.75 | 31.82 | 71.43   | 77.78 | 45.45     | 47.62      | 57.69   | 14.29 | 38.89 | 42.34                |
| Mplug-owl2-llama2-7B            | 38.10    | 17.65  | 22.22     | 61.54 | 25.00 | 57.14   | 76.92 | 40.00     | 47.37      | 65.22   | 26.32 | 46.15 | 42.86                |
| LLaVA-RLHF (7B)                 | 86.36    | 50.00  | 50.00     | 46.67 | 40.00 | 78.95   | 81.82 | 38.10     | 57.89      | 68.18   | 29.41 | 56.25 | 57.01                |

## 🚩 MMR-data

To enhance MLLMs' understanding capability and robustness, we propose a data construction method using GPT-4V to generate paired positive and negative samples for instruction tuning. The method includes three steps: 1) Information extraction. We implicitly and comprehensively extract detailed information from images, including text, object attributes, human characteristics, relationships between objects, relationships between people, events, and overall perception. 2) Instruction tuning data generation. We generate positive samples using the extracted information and construct negative samples that directly contradict the positive ones. 3) Sample filtering. We filter samples through keyword matching to remove those with uncertain answers and redundant phrases.

<p align="center">
  <img src="./data_collection.png" alt="Logo">
</p>

### Data generation
- Generate conversations based on GPT-4V
  
```shell
python dataset/data_generation.py \
      --input_file /path/to/input.json \
      --output_file /path/to/output.json \
      --image_folder /path/to/image folder \
      --api_key api_key
```

- Reformat the JSON

```shell
python dataset/data_reformat.py \
      --input /path/to/input.json \
      --output_pos /path/to/output_pos.json \
      --output_neg /path/to/output_neg.json \
      --output_merge /path/to/merged_output.json
```

- Filter the JSON

```shell
python dataset/data_filtering.py \
      --input /path/to/input.json \
      --output /path/to/output.json
```
  
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
