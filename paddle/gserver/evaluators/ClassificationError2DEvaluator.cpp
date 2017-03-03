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


#include "paddle/utils/Stat.h"
#include "paddle/gserver/evaluators/Evaluator.h"

#include "paddle/gserver/gradientmachines/NeuralNetwork.h"

P_DECLARE_int32(trainer_id);

namespace paddle {

class ClassificationError2DEvaluator : public Evaluator {
public:
  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    CHECK_EQ(arguments.size(), 3);
    CHECK_EQ(arguments[1].getBatchSize(), arguments[2].getBatchSize());
    numSamples_ += arguments[1].getBatchSize();
  }

  MatrixPtr calcError(std::vector<Argument>& arguments) {
    CHECK_EQ(arguments.size(), 3);
    IVectorPtr& output = arguments[0].ids;
    IVectorPtr& first = arguments[1].ids;
    IVectorPtr& second = arguments[2].ids;
    CHECK(output &&  first && second);
    const MatrixPtr errorMat = Matrix::create(first->getSize(),
        1, false, false);

    int *firstValue = first->getData();
    int *secondValue = second->getData();
    int *outputValue = output->getData();
    real *errorValue = errorMat->getData();
    errorMat->zeroMem();
    for (size_t i = 0; i < first->getSize(); ++i) {
      int pred_first = outputValue[2 * i];
      int pred_second = outputValue[2 * i + 1];
      errorValue[i] = pred_first != firstValue[i]
                      || pred_second != secondValue[i];
    }
    return errorMat;
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    MatrixPtr errorMat = calcError(arguments);
    return errorMat->getSum();
  }

  virtual void distributeEval(ParameterClient2* client) {
    mergeResultsOfAllClients(client);
  }
};

REGISTER_EVALUATOR(classification_error_2d,
                   ClassificationError2DEvaluator);
}  // namespace paddle
