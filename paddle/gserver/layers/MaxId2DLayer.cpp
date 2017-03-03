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

#include "Layer.h"

namespace paddle {

class MaxId2DLayer : public Layer {
public:
  explicit MaxId2DLayer(const LayerConfig& config) : Layer(config) {}

  virtual bool init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
    bool ret = Layer::init(layerMap, parameterMap);
    CHECK_EQ(1UL, inputLayers_.size());
    return ret;
  }

  virtual void forward(PassType passType) {
    Layer::forward(passType);
    const Argument& input = getInput(0);
    CHECK(input.sequenceStartPositions);
    CHECK(input.subSequenceStartPositions);
    CHECK_EQ(input.value->getWidth(), 1UL);

    auto startPositions =
        input.sequenceStartPositions->getVector(false);
    auto subStartPositions =
        input.subSequenceStartPositions->getVector(false);
    size_t numSequences = input.getNumSequences();
    const int* starts = startPositions->getData();
    const int* subStarts = subStartPositions->getData();

    real *inputValue = input.value->getData();

    IVector::resizeOrCreate(output_.ids, numSequences * 2, false);
    int *idsValue = output_.ids->getData();

    output_.value = nullptr;
    int k = 0;
    for (size_t i = 0; i < numSequences; ++i) {
      real maxValue = inputValue[starts[i]];
      int maxId = starts[i];
      for (int j = starts[i] + 1; j < starts[i + 1]; ++j) {
        if (inputValue[j] > maxValue) {
          maxValue = inputValue[j];
          maxId = j;
        }
      }
      while (subStarts[k] < starts[i])
        ++k;
      int s = k;
      while (subStarts[k] <= maxId)
        ++k;
      idsValue[i * 2] = k - 1 - s;
      idsValue[i * 2 + 1] = maxId - subStarts[k - 1];
    }
  }

  virtual void backward(const UpdateCallback& callback) {}
};

REGISTER_LAYER(maxid_2d, MaxId2DLayer);

}  // namespace paddle
