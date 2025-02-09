Codebase for NAACL 2025 (Findings) Paper: A Guide To Effectively Leveraging LLMs for Low-Resource Text
Summarization: Data Augmentation and Semi-supervised Approaches

## Setup

Run `pip install -r requirements.txt` to install the necessary dependencies

## Running experiments

Exps can be launched by the following two commands:

```bash
cd mixsumm; python -m src.trainval -e mixsumm -j 0 # for running MixSumm experiments
cd ppsl; python -m src.trainval -e ssl -j 0  # for running all PPSL experiments
```

You can tweak the hyperparameters in mixsumm/exp_configs/mixsumm_exps.py and ppsl/exp_configs/ppsl_exps.py to adjust the experiment settings.

## Citing this work
If you find our work useful, please cite us:

```bibtex
@misc{sahu2025guideeffectivelyleveragingllms,
      title={A Guide To Effectively Leveraging LLMs for Low-Resource Text Summarization: Data Augmentation and Semi-supervised Approaches}, 
      author={Gaurav Sahu and Olga Vechtomova and Issam H. Laradji},
      year={2025},
      eprint={2407.07341},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.07341}, 
}
```

## Acknowledgements
This codebase uses the https://github.com/HHousen/TransformerSum framework for training summarization models.