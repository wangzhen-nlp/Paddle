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
#include "TransformLayer.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

REGISTER_LAYER(transform, TransformLayer);

bool TransformLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap))
    return false;
  CHECK_EQ(inputLayers_.size(), 1UL);
  setNeedSequenceInfo(false);
  return true;
}

void TransformLayer::forward(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(inputLayers_.size(), 1U);

  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  CHECK(input.subSequenceStartPositions);

  auto startPositions =
      input.sequenceStartPositions->getVector(false);
  auto subStartPositions =
      input.subSequenceStartPositions->getVector(false);
  const int* starts = startPositions->getData();
  const int* subStarts = subStartPositions->getData();

  size_t numSequences = input.getNumSequences();
  CHECK_EQ(numSequences, startPositions->getSize() - 1);
  CHECK_EQ(input.getBatchSize(), starts[numSequences]);

  std::vector<int> outputStarts;
  std::vector<int> outputSubStarts;
  outputStarts.push_back(0);
  outputSubStarts.push_back(0);

  size_t j = 0;
  for (size_t i = 1; i <= numSequences; ++i) {
    size_t s = j;
    while (subStarts[j] < starts[i])
      ++j;
    CHECK_EQ(subStarts[j] - subStarts[j - 1], (int)(j - s));
    for (size_t k = 0; k < j - s; ++k)
      outputSubStarts.push_back(outputSubStarts.back() + (int)(j - s));
    outputStarts.push_back(outputSubStarts.back());
  }

  reserveOutput(outputStarts.back(), 1);
  std::vector<int> copyIds(input.getBatchSize());
  j = 0;
  for (size_t i = 1; i <= numSequences; ++i) {
    size_t s = j;
    for (; outputSubStarts[j] < outputStarts[i]; ++j) {
      for (int k = outputSubStarts[j]; k < outputSubStarts[j + 1]; ++k) {
        int b = j - s;
        int e = k - outputSubStarts[j];
        if (b > e)
          continue;
        copyIds[starts[i - 1] + e * (e + 1) / 2 + b] = k;
      }
    }
  }

  ICpuGpuVector::resizeOrCreate(output_.sequenceStartPositions,
                                outputStarts.size(), false);
  ICpuGpuVector::resizeOrCreate(output_.subSequenceStartPositions,
                                outputSubStarts.size(), false);
  output_.sequenceStartPositions->
      copyFrom(outputStarts.data(), outputStarts.size(), false);
  output_.subSequenceStartPositions->
      copyFrom(outputSubStarts.data(), outputSubStarts.size(), false);

  IVector::resizeOrCreate(copyIds_, copyIds.size(), false);
  copyIds_->copyFrom(copyIds.data(), copyIds.size());

  reserveOutput(outputStarts.back(), 1);
  output_.value->zeroMem();
  real* outputs = output_.value->getData();
  real* inputs = input.value->getData();
  for (int i = 0; i < input.getBatchSize(); ++i)
      outputs[copyIds[i]] = inputs[i];
}

void TransformLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  const MatrixPtr& inputGrad = input.grad;
  const MatrixPtr& outputGrad = getOutputGrad();
  inputGrad->selectRows(*outputGrad, *copyIds_);
}

}  // namespace paddle
