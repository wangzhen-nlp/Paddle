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


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "SequenceOneInstanceLayer.h"

namespace paddle {

REGISTER_LAYER(seqoneins, SequenceOneInstanceLayer);

bool SequenceOneInstanceLayer::init(const LayerMap& layerMap,
                                     const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(2U, inputLayers_.size());

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  // transform to which sequence type
  if (config_.trans_type() == "non-seq") {
    type_ = kNonSeq;
  } else if (config_.trans_type() == "seq") {
    type_ = kSeq;
  } else {
    LOG(FATAL) << "Unknown trans_type: " << config_.trans_type();
  }
  setNeedSequenceInfo(false);

  tmpSrc_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  tmpDest_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);

  return true;
}

void SequenceOneInstanceLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  const Argument& offset = getInput(1);
  newBatchSize_ = type_ ? input.getNumSubSequences() : input.getNumSequences();
  size_t dim = getSize();
  // check
  CHECK_EQ(dim, input.value->getWidth());
  auto offsetValue = offset.ids;
  CHECK_EQ(newBatchSize_, offsetValue->getSize());

  resetOutput(newBatchSize_, dim);
  if (type_) {
    CHECK(input.subSequenceStartPositions)
      << "when trans_type = seq, input must hasSubseq";
  }
  if (type_) {
    output_.degradeSequence(input, useGpu_);
  }

  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceOneInstanceLayerForward", getName().c_str());

    for (size_t seqId = 0; seqId < newBatchSize_; ++seqId) {
      int insId = offsetValue->getElement(seqId);

      outputValue->subMatrix(seqId, 1, tmpDest_)
          ->assign(*(inputValue->subMatrix(insId, 1, tmpSrc_)));
    }
  }

  if (biases_.get() != NULL) {
    outputValue->addBias(*(biases_->getW()), 1);
  }

  /*  activation, should set to 'linear' in most cases */
  forwardActivation();
}

void SequenceOneInstanceLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  const Argument& offset = getInput(1);
  auto offsetValue = offset.ids;
  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  size_t numSequences = offsetValue->getSize();

  if (inputGrad) {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceOneInstanceLayerBackward", getName().c_str());

    for (size_t seqId = 0; seqId < numSequences; ++seqId) {
      int insId = offsetValue->getElement(seqId);

      inputGrad->subMatrix(insId, 1, tmpDest_)
          ->add(*(outputGrad->subMatrix(seqId, 1, tmpSrc_)));
    }
  }
}

}  // namespace paddle
