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

#include <memory>
#include <vector>
#include "Layer.h"

namespace paddle {

class MultiClassCrossEntropy2D : public Layer {
public:
  explicit MultiClassCrossEntropy2D(const LayerConfig& config) :
      Layer(config) {}
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  LayerPtr getOutputLayer() { return inputLayers_[0]; }
  LayerPtr getFirstLabelLayer() { return inputLayers_[1]; }
  LayerPtr getSecondLabelLayer() { return inputLayers_[2]; }
  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

}  // namespace paddle
