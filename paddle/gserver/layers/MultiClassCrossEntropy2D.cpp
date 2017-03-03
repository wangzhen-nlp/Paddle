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
#include "MultiClassCrossEntropy2D.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

REGISTER_LAYER(multi_class_cross_entropy_2d,
               MultiClassCrossEntropy2D);

bool MultiClassCrossEntropy2D::init(const LayerMap& layerMap,
                                    const ParameterMap& parameterMap) {
  bool ret = Layer::init(layerMap, parameterMap);
  if (!ret) return ret;
  CHECK_EQ(inputLayers_.size(), 3UL);
  return true;
}

void MultiClassCrossEntropy2D::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& output = getInput(*getOutputLayer());
  CHECK(output.sequenceStartPositions);
  CHECK(output.subSequenceStartPositions);
  CHECK_EQ(output.value->getWidth(), 1UL);
  size_t numSequences1 = output.getNumSequences();
  auto startPositions1 =
      output.sequenceStartPositions->getVector(false);
  auto substartPositions1 =
      output.subSequenceStartPositions->getVector(false);

  const Argument& firstLabel = getInput(*getFirstLabelLayer());
  CHECK(firstLabel.sequenceStartPositions);
  size_t numSequences2 = firstLabel.getNumSequences();
  auto startPositions2 =
      firstLabel.sequenceStartPositions->getVector(false);

  const Argument& secondLabel = getInput(*getSecondLabelLayer());
  CHECK(secondLabel.sequenceStartPositions);
  size_t numSequences3 = secondLabel.getNumSequences();
  auto startPositions3 =
      secondLabel.sequenceStartPositions->getVector(false);

  CHECK_EQ(startPositions1->getData()[numSequences1],
           output.getBatchSize());
  CHECK_EQ(numSequences1, startPositions1->getSize() - 1);

  CHECK_EQ(startPositions2->getData()[numSequences2],
           firstLabel.getBatchSize());
  CHECK_EQ(numSequences2, startPositions2->getSize() - 1);

  CHECK_EQ(startPositions3->getData()[numSequences3],
           secondLabel.getBatchSize());
  CHECK_EQ(numSequences3, startPositions3->getSize() - 1);

  CHECK_EQ(numSequences1, numSequences2);
  CHECK_EQ(numSequences2, numSequences3);

  resetOutput(numSequences1, 1);

  MatrixPtr outputValue = output.value;
  IVectorPtr firstValue = firstLabel.ids;
  IVectorPtr secondValue = secondLabel.ids;

  CHECK_EQ(firstValue->getSize(), numSequences1);
  CHECK_EQ(secondValue->getSize(), numSequences1);

  const int* starts = startPositions1->getData();
  const int* substarts = substartPositions1->getData();
  const real* out = outputValue->getData();
  const int* first = firstValue->getData();
  const int* second = secondValue->getData();

  real* cost = getOutputValue()->getData();
  size_t j = 0;
  for (size_t i = 0; i < numSequences1; ++i) {
    while (substarts[j] < starts[i])
      ++j;
    j += first[i];
    int pos = substarts[j] + second[i];
    cost[i] = -std::log(out[pos]);
  }
}

void MultiClassCrossEntropy2D::backward(const UpdateCallback& callback) {
  (void)callback;
  const Argument& output = getInput(*getOutputLayer());
  const Argument& firstLabel = getInput(*getFirstLabelLayer());
  const Argument& secondLabel = getInput(*getSecondLabelLayer());

  size_t numSequences1 = output.getNumSequences();
  auto startPositions1 =
      output.sequenceStartPositions->getVector(false);
  auto substartPositions1 =
      output.subSequenceStartPositions->getVector(false);

  MatrixPtr outputValue = output.value;
  MatrixPtr gradValue = output.grad;
  IVectorPtr firstValue = firstLabel.ids;
  IVectorPtr secondValue = secondLabel.ids;

  const int* starts = startPositions1->getData();
  const int* substarts = substartPositions1->getData();
  const real* out = outputValue->getData();
  const int* first = firstValue->getData();
  const int* second = secondValue->getData();

  real* grad = gradValue->getData();
  size_t j = 0;
  for (size_t i = 0; i < numSequences1; ++i) {
    while (substarts[j] < starts[i])
      ++j;
    j += first[i];
    int pos = substarts[j] + second[i];
    grad[pos] -= 1 / out[pos];
  }
}

}  // namespace paddle
