/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <algorithm>
#include "paddle/utils/Logging.h"
#include <cmath>
#include "AggregateLayer.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

REGISTER_LAYER(aggregate, AggregateLayer);

bool AggregateLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  bool ret = Layer::init(layerMap, parameterMap);
  if (!ret) return ret;
  CHECK_EQ(inputLayers_.size(), 1UL);
  return true;
}

void AggregateLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  CHECK(input.subSequenceStartPositions);
  CHECK_EQ(input.value->getWidth(), 1UL);

  auto positions =
      input.sequenceStartPositions->getVector(false);
  auto subPositions =
      input.subSequenceStartPositions->getVector(false);
  size_t numSequences = input.getNumSequences();
  CHECK_EQ(numSequences, positions->getSize() - 1);

  const int* starts = positions->getData();
  const int* subStarts = subPositions->getData();
  const real* inputs = input.value->getData();

  reserveOutput(input.getBatchSize(), 1);
  real* outputs = output_.value->getData();
  size_t j = 0;
  for (size_t i = 1; i <= numSequences; ++i) {
    std::vector<real> res;
    for (; subStarts[j] < starts[i]; ++j) {
      for (int k = subStarts[j]; k < subStarts[j + 1]; ++k) {
        if (k - subStarts[j] >= (int)res.size())
          res.push_back(1);
        CHECK(inputs[k]);
        res[k - subStarts[j]] *= inputs[k];
        outputs[k] = res[k - subStarts[j]];
      }
    }
    CHECK_EQ(subStarts[j], starts[i]);
  }
}

void AggregateLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  auto positions =
      input.sequenceStartPositions->getVector(false);
  auto subPositions =
      input.subSequenceStartPositions->getVector(false);
  size_t numSequences = input.getNumSequences();
  const int* starts = positions->getData();
  const int* subStarts = subPositions->getData();
  const real* inputs = input.value->getData();
  const real* outputs = output_.value->getData();

  real* inputGrad = getInputGrad(0)->getData();
  real* outputGrad = getOutputGrad()->getData();

  size_t j = 0;
  for (size_t i = 1; i <= numSequences; ++i) {
    std::vector<real> res;
    size_t s = j;
    while (subStarts[s] < starts[i])
      ++s;
    CHECK_EQ(subStarts[s], starts[i]);
    for (j = s - 1; subStarts[j] >= starts[i - 1]; --j) {
      for (int k = subStarts[j]; k < subStarts[j + 1]; ++k) {
        if (k - subStarts[j] >= (int)res.size())
          res.push_back(0);
        res[k - subStarts[j]] += outputGrad[k] * outputs[k];
        CHECK(inputs[k]);
        inputGrad[k] = res[k - subStarts[j]] / inputs[k];
      }
      if (subStarts[j] == 0)
        break;
    }
    j = s;
  }
}

}  // namespace paddle
