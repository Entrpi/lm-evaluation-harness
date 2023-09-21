"""
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/pdf/2304.06364.pdf
his benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for
general human test-takers, such as general college admission tests
(e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions,
lawyer qualification tests, and national civil service exams.
Homepage: https://github.com/microsoft/AGIEval/tree/main
"""
from lm_eval.base import MultipleChoiceTask
import numpy as np
import datasets


# TODO: Add the BibTeX citation for the task.
_CITATION = """
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models}, 
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@inproceedings{ling-etal-2017-program,
    title = "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems",
    author = "Ling, Wang  and
      Yogatama, Dani  and
      Dyer, Chris  and
      Blunsom, Phil",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1015",
    doi = "10.18653/v1/P17-1015",
    pages = "158--167",
    abstract = "Solving algebraic word problems requires executing a series of arithmetic operations{---}a program{---}to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.",
}
@inproceedings{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
@inproceedings{Liu2020LogiQAAC,
  title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
  author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2020}
}
@inproceedings{zhong2019jec,
  title={JEC-QA: A Legal-Domain Question Answering Dataset},
  author={Zhong, Haoxi and Xiao, Chaojun and Tu, Cunchao and Zhang, Tianyang and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of AAAI},
  year={2020},
}
@article{Wang2021FromLT,
  title={From LSAT: The Progress and Challenges of Complex Reasoning},
  author={Siyuan Wang and Zhongkun Liu and Wanjun Zhong and Ming Zhou and Zhongyu Wei and Zhumin Chen and Nan Duan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021},
  volume={30},
  pages={2201-2216}
}
"""


# TODO: Replace `NewTask` with the name of your Task.
class AgiEvalTask(MultipleChoiceTask):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "dmayhem93/agieval-"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self, task_name, data_dir=None, cache_dir=None, download_mode=None):
        self.task_name = task_name
        super().__init__(data_dir, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH + self.task_name,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": doc['query'],  # The query prompt.
            "choices": doc['choices'],  # The list of choices.
            "gold": doc['gold'],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if int(np.argmax(results)) in gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if int(np.argmax(results / completion_len)) in gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }


def create_task_from_split(split):
    class WrappedTask(AgiEvalTask):
        def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
            super().__init__(split, data_dir, cache_dir, download_mode)

    return WrappedTask


def create_all_tasks():
    supported_tasks = [
        "aqua-rat",
        "gaokao-biology",
        "gaokao-chemistry",
        "gaokao-chinese",
        "gaokao-english",
        "gaokao-geography",
        "gaokao-history",
        "gaokao-mathqa",
        "gaokao-physics",
        "logiqa-en",
        "logiqa-zh",
        "lsat-ar",
        "lsat-lr",
        "lsat-rc",
        "sat-en",
        "sat-en-without-passage",
        "sat-math",
    ]
    res = {}
    for task_name in supported_tasks:
        res[f"agieval_{task_name.replace('-', '_')}"] = create_task_from_split(task_name)
    return res