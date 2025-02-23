# DINER: Debiasing Aspect-based Sentiment Analysis with Multi-variable Causal Inference

## Overall

The SCM of ABSA, which is formulated as a directed acyclic graph, is shown in (a). With the SCM defined, we can derive the formula of causal effect.
As shown in (b), the desired situation for ABSA is that the edges that bring biases are all blocked.

<p align="center"><img src='./assets/SCM.png'  width=500> </p>
We present a novel debiasing framework, DINER, for multi-variable causal inference. 
<p align="center"><img src='./assets/method.png'  width=500> </p>

## Requirements
```
conda create -n diner python=3.10
conda activate diner
pip install -r requirements.txt
```

## Run
Our experiments are carried out with an NVIDIA A100 80GB GPU.
```
cd src
bash run_diner.sh ${dataset_name}
```

## 🌻 Acknowledgement
This work is implemented by [ARTS](https://github.com/zhijing-jin/ARTS_TestSet), [cfvqa](https://github.com/yuleiniu/cfvqa), and [CCD](https://github.com/farewellthree/Causal-Context-Debiasing). Sincere thanks for their efforts.

## 📖Citation

```bibtex
@misc{wu2024diner,
    title={DINER: Debiasing Aspect-based Sentiment Analysis with Multi-variable Causal Inference},
    author={Jialong Wu and Linhai Zhang and Deyu Zhou and Guoqiang Xu},
    year={2024},
    eprint={2403.01166},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
