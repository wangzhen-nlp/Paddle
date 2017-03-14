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
#include "IdentityCostLayer.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

REGISTER_LAYER(identity_cost,
               IdentityCostLayer);

bool IdentityCostLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  bool ret = Layer::init(layerMap, parameterMap);
  if (!ret) return ret;
  CHECK_EQ(inputLayers_.size(), 1UL);
  return true;
}

void IdentityCostLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& output = getInput(*getOutputLayer());
  auto height = output.value->getHeight();
  auto width = output.value->getWidth();
  resetOutput(height, width);

  MatrixPtr outputValue = output.value;
  const real* out = outputValue->getData();
  real* cost = getOutputValue()->getData();
  for (size_t i = 0; i < height * width; ++i)
      cost[i] = -std::log(out[i]);
}

void IdentityCostLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  const Argument& output = getInput(*getOutputLayer());
  auto height = output.value->getHeight();
  auto width = output.value->getWidth();

  MatrixPtr outputValue = output.value;
  MatrixPtr gradValue = output.grad;
  const real* out = outputValue->getData();
  real* grad = gradValue->getData();
  for (size_t i = 0; i < height * width; ++i)
    grad[i] -= 1 / out[i];
}

}  // namespace paddle
