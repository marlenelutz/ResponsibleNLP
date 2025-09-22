# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

#import torch
from robbie.datasets import Prompt
from robbie.predictors._base import Prediction, Predictor
#from robbie.utils import batch_iter
from vllm import LLM
from transformers import AutoTokenizer


def _isset(value):
    return value is not None and value > 0


class VLLMPredictor(Predictor):
    @classmethod
    def add_args(cls, parser):
        # Register any required command line args
        parser.add_argument("--model_id", type=str)
        parser.add_argument("--num_gpus", default=1, type=int)
        #parser.add_argument("--max_model_len", default=10000, type=int)
        parser.add_argument("--max_new_tokens", default=100, type=int)
        #parser.add_argument("--device", type=str, default="cpu")
        return parser

    @classmethod
    def from_args(cls, args):
        # Build the predictor from captured args
        return VLLMPredictor(
            model_id=args.model_id,
            num_gpus=args.num_gpus,
            #max_model_len = args.max_model_len,
            max_new_tokens = args.max_new_tokens,
            seed = args.seed,
        )

    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        #max_model_len: int,
        max_new_tokens: int, # self.max_new_tokens
        seed: int,
    ):
        self.model_id = model_id
        self.model = LLM(model=self.model_id, 
                         generation_config="auto",
                         tensor_parallel_size=num_gpus,
                         #max_model_len=max_model_len
                        )
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.num_gpus = num_gpus
        #self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        

    @property
    def name(self):
        return self.model_id.replace("/", "_")

    #def _generation_kwargs(self):
    #    gen_args = {
    #        "top_k": self.config.top_k,
    #        "top_p": self.config.top_p,
    #        "temperature": self.config.temperature,
    #        "num_beams": self.config.beam_size,
    #        "max_new_tokens": self.config.max_length,
    #        "do_sample": (
    #            _isset(self.config.top_k)
    #            or _isset(self.config.top_p)
    #            or _isset(self.config.temperature)
    #        ),
    #    }
    #    return {k: v for k, v in gen_args.items() if v is not None}

    #@torch.inference_mode()
    def generate(self, prompts: Iterator[Prompt]) -> Iterator[Prediction]:
        sampling_params = self.model.get_default_sampling_params()

        # generation config
        sampling_params.max_new_tokens = self.max_new_tokens
        sampling_params.seed = self.seed

        # construct the prompts
        prompts = list(prompts)
        messages = [[{"role": "user", "content": f"Continue the following text: {prompt.text}"}] for prompt in prompts]
        chats = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        outputs = self.model.generate(chats, sampling_params, use_tqdm=True)

        predictions = (Prediction(prompt=prompt.text, generation=f"{prompt.text} {gen.outputs[0].text}", meta=prompt.meta) for prompt, gen in zip(prompts, outputs))
        return predictions



Predictor.register(
    name="vllm_chat",
    factory=VLLMPredictor.from_args,
    add_args=VLLMPredictor.add_args,
)
