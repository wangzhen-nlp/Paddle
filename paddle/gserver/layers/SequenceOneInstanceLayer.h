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

namespace paddle {

class SequenceOneInstanceLayer : public Layer {
protected:
  int type_;
  std::unique_ptr<Weight> biases_;
  enum SequenceLevel { kNonSeq = 0, kSeq = 1 };
  size_t newBatchSize_;
  ICpuGpuVectorPtr startPositions_;

  MatrixPtr tmpSrc_;
  MatrixPtr tmpDest_;

public:
  explicit SequenceOneInstanceLayer(const LayerConfig& config) :
    Layer(config) {}

  virtual ~SequenceOneInstanceLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

}  // namespace paddle
