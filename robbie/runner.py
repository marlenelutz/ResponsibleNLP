# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import asdict
from typing import Optional

from robbie.datasets import Dataset, Dataset
from robbie.metrics import Metric
from robbie.predictors import Predictor


class Runner:
    def __init__(
        self,
        dataset: Dataset,
        predictor: Predictor,
        metric: Metric,
        result_dir: str,
        num_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.predictor = predictor
        self.metric = metric
        self.num_samples = num_samples
        self.log_path = os.path.join(
            result_dir,
            f"scores.d__{dataset.name}.p__{predictor.name}.m__{metric.name}.json",
        )

        os.makedirs(result_dir, exist_ok=True)

    def run(self):
        #predictions = self.predictor.generate(self.dataset)
        #if self.num_samples and len(predictions) >= self.num_samples:
        #    predictions = predictions[:self.num_samples]

        predictions = []

        for p in self.predictor.generate(self.dataset):
            predictions.append(p)
            if self.num_samples and len(predictions) >= self.num_samples:
                break
        print(predictions[:2])

        result = self.metric.score(p for p in predictions)

        with open(self.log_path, "w") as f:
            f.write(json.dumps(asdict(result), indent=2))
