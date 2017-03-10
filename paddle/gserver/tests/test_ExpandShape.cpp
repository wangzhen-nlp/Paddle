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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/math/MathUtils.h"

#include "LayerGradUtil.h"
#include "TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_bool(use_gpu);
P_DECLARE_int32(gpu_id);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);

void testExpandShapeLayer(string trans_type) {
  TestConfig config;
  config.layerConfig.set_type("expand_shape");
  config.inputDefs.push_back(
      {trans_type == "non-seq" ? INPUT_DENSE_DIM_DATA : INPUT_SEQUENCE_DATA,
       "layer_0", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.set_trans_type(trans_type);
  LOG(INFO) << " trans_type=" << trans_type;

  for (auto useGpu : {true, false}) {
    testLayerGrad(config, "expand_shape", 30, false, useGpu);
  }
}

TEST(Layer, ExpandShapeLayer) {
  testExpandShapeLayer("non-seq");
  testExpandShapeLayer("seq");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
