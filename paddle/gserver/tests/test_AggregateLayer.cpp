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

void initDataLayer2(TestConfig testConf, std::vector<DataLayerPtr>* dataLayers,
                   vector<Argument>* datas, LayerMap* layerMap,
                   string testLayerName, size_t batchSize, bool trans,
                   bool useGpu) {
  ICpuGpuVectorPtr sequenceStartPositions;
  ICpuGpuVectorPtr subSequenceStartPositions;
  CHECK_EQ(testConf.inputDefs.size(), 1);
  CHECK_EQ(testConf.inputDefs[0].inputType, INPUT_HASSUB_SEQUENCE_DATA);
  CHECK_EQ(testConf.inputDefs[0].dim, 1);

  LayerConfig config;
  config.set_name(testConf.inputDefs[0].name);
  config.set_type("data");
  config.set_size(testConf.inputDefs[0].dim);
  LayerPtr layer = LayerPtr(new DataLayer(config));

  vector<int> starts;
  vector<int> subStarts;
  starts.push_back(0);
  subStarts.push_back(0);
  for (size_t i = 0; i < batchSize; ++i) {
    int seqLen = uniformRandom(10) + 1;
    for (int j = 1; j <= seqLen; ++j)
      subStarts.push_back(subStarts.back() + j);
    starts.push_back(subStarts.back());
  }
  ICpuGpuVector::resizeOrCreate(sequenceStartPositions,
      starts.size(), useGpu);
  ICpuGpuVector::resizeOrCreate(subSequenceStartPositions,
      subStarts.size(), useGpu);
  sequenceStartPositions->
      copyFrom(starts.data(), starts.size(), useGpu);
  subSequenceStartPositions->
      copyFrom(subStarts.data(), subStarts.size(), useGpu);

  Argument data;
  data.value = Matrix::create(starts.back(), 1, false, useGpu);
  data.grad = Matrix::create(starts.back(), 1, false, useGpu);
  data.value->randomizeUniform();
  data.value->add(-0.5);
  data.value->sigmoid(*data.value);
  data.grad->zeroMem();

  data.sequenceStartPositions = sequenceStartPositions;
  data.subSequenceStartPositions = subSequenceStartPositions;

  DataLayerPtr dataLayer = std::dynamic_pointer_cast<DataLayer>(layer);
  dataLayer->setData(data);
  dataLayer->forward(PASS_GC);
  dataLayers->push_back(dataLayer);
  (*layerMap)[config.name()] = layer;
  datas->push_back(data);
}

void testLayerGradKernel2(TestConfig testConf, string testLayerName,
                         size_t batchSize, bool trans, bool useGpu,
                         bool useWeight = false, float epsilon = 0.02) {
#ifdef PADDLE_ONLY_CPU
  if (useGpu) return;
#endif
  FLAGS_use_gpu = useGpu;
  FLAGS_prev_batch_state = testConf.testBatchState;
  MatrixPtr weights = nullptr;
  testConf.layerConfig.set_name(testLayerName);
  LOG(INFO) << " layer_type=" << testConf.layerConfig.type()
            << " useGpu=" << useGpu;

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer2(testConf, &dataLayers, &datas, &layerMap, testLayerName,
                batchSize, trans, useGpu);
  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr testLayer;
  initTestLayer(testConf, &layerMap, &parameters, &testLayer);

  LayerStatePtr state = std::make_shared<LayerState>();
  if (testConf.testBatchState) {
    initBatchState(dataLayers[0], testLayer, state, useGpu);
    testLayer->resetState();
    testLayer->setState(state);
  }

  testLayer->forward(PASS_GC);
  if (useWeight && weights == nullptr) {
    weights = testLayer->getOutput().value->clone(0, 0, useGpu);
    initWeight(weights);
  }
  std::vector<Argument> outArgs;
  outArgs.push_back(testLayer->getOutput());
  if (useWeight) {
    outArgs[0].value = outArgs[0].value->clone(0, 0, useGpu);
    outArgs[0].value->dotMul(*testLayer->getOutput().value, *weights);
  }

  real cost = Argument::sumCosts(outArgs);
  LOG(INFO) << " cost " << cost;
  EXPECT_FALSE(std::isnan(cost));

  // Test whether the callback is called for a parameter
  if (testLayer->getOutputGrad()) {
    useWeight ? testLayer->getOutput().grad->copyFrom(*weights)
              : testLayer->getOutputGrad()->resetOne();
  }
  vector<int> callbackFlags(parameters.size(), 0);
  auto callback = [&](Parameter* para) { ++callbackFlags[para->getID()]; };
  testLayer->backward(callback);

  // do forward and backward for another time to test that gradient is doubled
  int callbackCount = 1;
  if (testConf.testAccumulate) {
    if (testConf.testBatchState) {
      testLayer->setState(state);
    }
    testLayer->forward(PASS_GC);
    if (testLayer->getOutputGrad()) {
      useWeight ? testLayer->getOutput().grad->copyFrom(*weights)
                : testLayer->getOutputGrad()->resetOne();
    }
    testLayer->backward(callback);
    ++callbackCount;
  }
  for (size_t i = 0; i < parameters.size(); ++i) {
    EXPECT_EQ(parameters[i]->isStatic() ? 0 : callbackCount,
              callbackFlags[i]);
  }

  // Test whether the layer's forward calculation is stable
  // by adding perturbation to its parameters or its input layers
  real maxDiff = 0;
  testPerturbParameter(testConf, weights, state, cost, callbackCount, &maxDiff,
                       testLayer, &parameters);
  testPerturbInput(testConf, weights, state, cost, callbackCount, &maxDiff,
                   testLayer, dataLayers);
  EXPECT_LE(fabs(maxDiff), epsilon);

  if (testConf.testState) {
    testState(testLayer, dataLayers, datas);
  }
  if (testConf.testBatchState) {
    testBatchState(testLayer, dataLayers, datas);
  }
}

void testLayerGrad2(TestConfig testConf, string testLayerName, size_t batchSize,
                   bool trans, bool useGpu, bool useWeight = false,
                   float epsilon = 0.02) {
  testLayerGradKernel2(testConf, testLayerName, batchSize, trans, useGpu,
                      useWeight, epsilon);
  bool isStaticTest = false;
  LayerConfig testConfig = testConf.layerConfig;
  for (size_t i = 0; i < testConf.inputDefs.size(); i++) {
    InputDef inputDef = testConf.inputDefs[i];
    // Some layer must set isStatic true, like DataNormLayer
    // so use !isStatic in if
    if (inputDef.paraSize && (!inputDef.isStatic)) {
      testConf.inputDefs[i].isStatic = true;
      isStaticTest = true;
    }
  }

  if (testConf.biasSize) {
    testConf.staticBias = true;
    isStaticTest = true;
  }
  if (isStaticTest) {
    testLayerGradKernel2(testConf, testLayerName, batchSize, trans, useGpu,
                        useWeight, epsilon);
  }
}

TEST(Layer, AggregateLayer) {
  TestConfig config;
  config.layerConfig.set_type("aggregate");
  config.inputDefs.push_back({INPUT_HASSUB_SEQUENCE_DATA, "layer_0", 1, 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false}) {
    testLayerGrad2(config, "aggregate", 10, false, useGpu);
  }
}

TEST(Layer, TransformLayer) {
  TestConfig config;
  config.layerConfig.set_type("transform");
  config.inputDefs.push_back({INPUT_HASSUB_SEQUENCE_DATA, "layer_0", 1, 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false}) {
    testLayerGrad2(config, "transform", 10, false, useGpu);
  }
}

void testPaddingLayer(string trans_type) {
  TestConfig config;
  config.layerConfig.set_type("padding");
  config.inputDefs.push_back({INPUT_HASSUB_SEQUENCE_DATA, "layer_0", 1, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.set_trans_type(trans_type);
  LOG(INFO) << " trans_type=" << trans_type;
  for (auto useGpu : {false}) {
    testLayerGrad2(config, "padding", 10, false, useGpu);
  }
}

TEST(Layer, PaddingLayer) {
  testPaddingLayer("non-seq");
  testPaddingLayer("seq");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
