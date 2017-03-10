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

#include "ExpandShapeLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(expand_shape, ExpandShapeLayer);

bool ExpandShapeLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 1UL);
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  // which sequence type of input[0]
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

void ExpandShapeLayer::forward(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(1U, inputLayers_.size());
  const Argument& dataInput = getInput(0);
  size_t outputBatchSize = dataInput.getBatchSize();

  if (type_) {
    CHECK(dataInput.sequenceStartPositions);
    CHECK(!dataInput.subSequenceStartPositions);
    output_.sequenceStartPositions =
        dataInput.sequenceStartPositions;
    output_.subSequenceStartPositions =
        dataInput.sequenceStartPositions;
  } else {
    CHECK(!dataInput.sequenceStartPositions);
    ICpuGpuVector::resizeOrCreate(output_.sequenceStartPositions,
                                  outputBatchSize + 1, false);
    int* starts =
        output_.sequenceStartPositions->getMutableData(false);
    for (size_t i = 0; i <= outputBatchSize; ++i)
      starts[i] = i;
  }

  // reserve output: Expand output to batchsize of sequence data.
  reserveOutput(outputBatchSize, dataInput.value->getWidth());

  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();
  outputValue->copyFrom(*inputValue);
  if (biases_.get() != NULL) {
    outputValue->addBias(*(biases_->getW()), 1);
  }
}

void ExpandShapeLayer::backward(const UpdateCallback& callback) {
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  AsyncGpuBlock asyncGpuBlock;
  inputGrad->add(*outputGrad);
}

}  // namespace paddle
