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
#include "PaddingLayer.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

REGISTER_LAYER(padding, PaddingLayer);

bool PaddingLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 1UL);
  if (config_.trans_type() == "non-seq") {
    type_ = kNonSeq;
  } else if (config_.trans_type() == "seq") {
    type_ = kSeq;
  } else {
    LOG(FATAL) << "Unknown trans_type: " << config_.trans_type();
  }
  setNeedSequenceInfo(false);
  return true;
}

void PaddingLayer::forward(PassType passType) {
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

  std::vector<int> copyIds;
  std::vector<int> outputStarts;
  std::vector<int> outputSubStarts;
  outputStarts.push_back(0);
  outputSubStarts.push_back(0);
  size_t j = 0;
  int offset = 0;
  if (type_) {
    for (size_t i = 1; i <= numSequences; ++i) {
      for (; subStarts[j] < starts[i]; ++j) {
        outputSubStarts.push_back(subStarts[j + 1] + offset);
        for (int k = subStarts[j]; k < subStarts[j + 1]; ++k)
          copyIds.push_back(k + offset);
      }
      CHECK_EQ(subStarts[j], starts[i]);
      offset += subStarts[j] - subStarts[j - 1] + 1;
      outputStarts.push_back(starts[i] + offset);
      outputSubStarts.push_back(starts[i] + offset);
    }
  } else {
    for (size_t i = 1; i <= numSequences; ++i) {
      ++offset;
      outputSubStarts.push_back(subStarts[j] + offset);
      for (; subStarts[j] < starts[i]; ++j, ++offset) {
        outputSubStarts.push_back(subStarts[j + 1] + offset + 1);
        for (int k = subStarts[j]; k < subStarts[j + 1]; ++k)
          copyIds.push_back(k + offset);
      }
      CHECK_EQ(subStarts[j], starts[i]);
      outputStarts.push_back(starts[i] + offset);
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

  CHECK_EQ(outputStarts.back(), outputSubStarts.back());

  IVector::resizeOrCreate(copyIds_, copyIds.size(), false);
  copyIds_->copyFrom(copyIds.data(), copyIds.size());

  reserveOutput(outputStarts.back(), 1);
  output_.value->resetOne();
  real* outputs = output_.value->getData();
  real* inputs = input.value->getData();
  for (int i = 0; i < input.getBatchSize(); ++i)
      outputs[copyIds[i]] = inputs[i];
}

void PaddingLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  const MatrixPtr& inputGrad = input.grad;
  const MatrixPtr& outputGrad = getOutputGrad();
  inputGrad->selectRows(*outputGrad, *copyIds_);
}

}  // namespace paddle
