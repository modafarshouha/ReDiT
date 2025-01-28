# ReDiT
[ReDiT: Re‑evaluating large visual question answering model confidence by defining input scenario Difficulty and applying Temperature mapping](https://link.springer.com/article/10.1007/s00530-024-01629-w) <br>

## Acknowledgement
The visual question answering models (and their weights) are from [Huggingface](https://huggingface.co/).

## Abstract
Large models (LMs) have achieved remarkable results in vision-language tasks. Such models are trained on vast amount of data then fine-tuned for downstream tasks like visual question answering (VQA). This wide exposure to data along with the complexity in a multi-modal setup (e.g. VQA) demand formalizing an extended definition of what constitutes out-of-distribution (OOD) condition for these models. Moreover, the input difficulty is expected to influence the model's performance, and it should be reflected on its confidence scoring. In this work, we primarily address large visual question answering (LVQA) models. We extend the classical boundaries of OOD definition and introduce a novel customizable dataset that simulates various challenges for LVQA models; i.e. 3U-VQA dataset. Moreover, we present a categorical scale to assess the input scenario difficulty. This scale is used to improve the reliability of the answer confidence score by re-evaluating it through adjusting a temperature parameter in the softmax function. Lastly, we study the credibility of our categorization and show that our re-evaluating method assists in reducing the overlap between correct and incorrect LVQA model predictions' scores.

## Description
### ReDiT
For a certain LVQA model *M*, given an image *I* and a text/question *Q*, we build our method by assuming two practical conditions; (1) the user is able to define the input scenario difficulty and (2) has access to the logits set *X*. Aiming for more objectivity when assessing the input scenario difficulty, we present a structured categorical method. Subsequently, we propose an approach to re-evaluate the model's answer confidence score with respect to the input scenario difficulty. This re-evaluation aims to provide more accurate scoring, particularly in cases where the answer's probability is interpreted as confidence.

When working with LVQA model within a post-hoc framework, we attribute the difficulty of any input scenario to external factor(s). The overall complexity of the input scenario arises from two sources; i.e. text and/or image. For instance, a challenging scenario may result from an image with ambiguous content, a question with unclear language, or a combination of both. Consequently, we categorize the criteria along two dimensions; i.e. source (image-related, question-related and their combination) and importance (major and minor). Although our main focus in this work are LVQA models, this set of criteria can be applied to any multi-modal framework, depending on the task objective; e.g. VQA or IC.

### 3U-VQA dataset 
To tackle the challenge of obtaining out-of-distribution (OOD) data for LVQA models, we introduce a novel dataset named 3U-VQA dataset (Usual, Unusual and Unknown object scenarios for LVQA with difficulty scoring dataset). The dataset comprises question and image sets. Each instance in the questions set is associated with a set of features representing the question-related criteria set. The questions and their ground truth answers are written using placeholders for the objects and their features, which can be specified based on the user needs and requirements. When creating the questions, we avoided deliberately binary (Yes/No) questions to prevent potential bias caused by the question's type in the model response.
<br>
The used dataset is available in <a href="./data/3U-VQA dataset">data folder</a>. <br><br>

## Code, experiments and results
VQA class and utils can be found in <a href="./src"> src folder</a>, while the experiments and results notebooks are in <a href="./results"> results folder</a>. <br>

Please refer to this repository and cite our publication when using this work.

## Cite
```
@Article{Al-Shouha2025,
author={Al-Shouha, Modafar
and Sz{\H{u}}cs, G{\'a}bor},
title={{ReDiT}: re-evaluating large visual question answering model confidence by defining input scenario difficulty and applying temperature mapping},
journal={Multimedia Systems},
year={2025},
month={Jan},
day={06},
volume={31},
number={1},
pages={45},
issn={1432-1882},
doi={10.1007/s00530-024-01629-w},
url={https://doi.org/10.1007/s00530-024-01629-w}
}
```

## License
This work license is: <a href="./Licenses/LICENSE">GNU Affero General Public License v3.0</a>. <br>
Please consider other works/models licenses when using them.
