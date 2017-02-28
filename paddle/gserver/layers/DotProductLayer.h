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


#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {
/**
 * @brief A layer for calculating dot product between two vector
 * \f[
 * f(x,y)=scale\times{x_1y_1+x_2y_2+...+x_ny_n}
 * \f]
 *
 * - Input1: A vector (batchSize * dataDim) *
 * - Input2: A vector (batchSize * dataDim) or (1 * dataDim) *
 * - Output: A vector (dataDim * 1)
 *
 * The config file api is dot_product.
 */
class DotProductLayer : public Layer {
public:
  explicit DotProductLayer(const LayerConfig& config)
      : Layer(config) {}

  ~DotProductLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

}  // namespace paddle
