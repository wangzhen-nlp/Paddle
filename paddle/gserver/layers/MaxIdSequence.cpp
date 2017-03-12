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

class MaxIdSequence : public Layer {
public:
  explicit MaxIdSequence(const LayerConfig& config) : Layer(config) {}

  virtual bool init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
    bool ret = Layer::init(layerMap, parameterMap);
    CHECK_EQ(2UL, inputLayers_.size());
    return ret;
  }

  virtual void forward(PassType passType) {
    Layer::forward(passType);
    const Argument& first = getInput(0);
    const Argument& second = getInput(1);
    CHECK(first.sequenceStartPositions);
    CHECK(second.sequenceStartPositions);
    CHECK(!first.subSequenceStartPositions);
    CHECK(!second.subSequenceStartPositions);
    CHECK_EQ(first.value->getWidth(), 1UL);
    CHECK_EQ(second.value->getWidth(), 1UL);

    auto firstPositions =
        first.sequenceStartPositions->getVector(false);
    auto secondPositions =
        second.SequenceStartPositions->getVector(false);
    size_t numSequences = first.getNumSequences();
    CHECK_EQ(numSequences, second.getNumSequences());
    const int* firstStarts = firstPositions->getData();
    const int* secondStarts = secondPositions->getData();
    
    real *firstValue = first.value->getData();
    real *secondValue = second.value->getData();

    IVector::resizeOrCreate(output_.ids, numSequences * 2, false);
    int *idsValue = output_.ids->getData();

    output_.value = nullptr;
    for (size_t i = 0; i < numSequences; ++i) {
      int seqLen = firstStarts[i + 1] - firstStarts[i];
      CHECK_EQ(seqLen, secondStarts[i + 1] - secondStarts[i]);
      int *maxFirstId = new int[seqLen];
      real *maxFirstVal = new real[seqLen];
      maxFirstId[0] = 0;
      maxFirstVal[0] = firstVal[firstStarts[i]];
      for (int j = 1; j < seqLen; ++j) {
        if (firstVal[firstStarts[i] + j] > maxFirstVal[j - 1]) {
          maxFirstId[j] = j;
          maxFirstVal[j] = firstVal[firstStarts[i] + j];
        } else {
          maxFirstId[j] = maxFirstId[j - 1];
          maxFirstVal[j] = maxFirstVal[j - 1];
        }
      }

      int *maxSecondId = new int[seqLen];
      real *maxSecondVal = new real[seqLen];
      maxSecondId[seqLen - 1] = seqLen - 1;
      maxSecondVal[seqLen - 1] = secondVal[firstStarts[i + 1] - 1];
      for (int j = seqLen - 2; j >= 0; --j) {
        if (secondVal[firstStarts[i] + j] > maxSecondVal[j + 1]) {
          maxSecondId[j] = j;
          maxSecondVal[j] = secondVal[firstStarts[i] + j];
        } else {
          maxSecondId[j] = maxSecondId[j + 1];
          maxSecondVal[j] = maxSecondVal[j + 1];
        }
        maxFirstVal[j] *= maxSecondVal[j];
      }

      int maxVal = maxFirstVal[0];
      int maxId = 0;
      for (int j = 1; j < seqLen; ++j) {
        if (maxFristVal[j] > maxVal) {
          maxVal = maxFirstVal[j];
          maxId = j;
        }
      }
      idsValue[i * 2] = maxFirstId[j];
      idsValue[i * 2 + 1] = maxSecondId[j];
      delete [] maxFirstId;
      delete [] maxFirstVal;
      delete [] maxSecondId;
      delete [] maxSecondVal;
    }
  }

  virtual void backward(const UpdateCallback& callback) {}
};

REGISTER_LAYER(maxid_seq, MaxIdSequence);

}  // namespace paddle
