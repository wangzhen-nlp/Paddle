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
#include <vector>

namespace paddle {

class Sequence {
private:
  std::vector<real> heap;
  std::vector<size_t> idxs;
  std::vector<size_t> invertIdxs;
  std::vector<real> res;
  std::vector<size_t> starts;
  std::vector<size_t> ends;
  size_t L;
  size_t K;
  size_t N;

  void swap(size_t aIdx, size_t bIdx) {
    {
      real tmp = heap[aIdx];
      heap[aIdx] = heap[bIdx];
      heap[bIdx] = tmp;
    }
    {
      idxs[invertIdxs[aIdx] - N] = bIdx;
      idxs[invertIdxs[bIdx] - N] = aIdx;
    }
    {
      size_t tmp = invertIdxs[aIdx];
      invertIdxs[aIdx] = invertIdxs[bIdx];
      invertIdxs[bIdx] = tmp;
    }
  }

public:
  Sequence(size_t _L, size_t _K) : L(_L), K(_K), N(0) {}

  ~Sequence() {
    heap.clear();
    idxs.clear();
    invertIdxs.clear();
    res.clear();
    starts.clear();
    ends.clear();
  }

  size_t getStarts(size_t idx) { return starts[idx]; }

  size_t getEnds(size_t idx) { return ends[idx]; }

  void adjustUpward(size_t idx) {
    if (!idx) return;
    size_t pIdx = (idx - 1) / 2;
    if (heap[pIdx] >= heap[idx]) return;
    swap(idx, pIdx);
    adjustUpward(pIdx);
  }

  void adjustDownward(size_t idx) {
    size_t lIdx = idx * 2 + 1;
    size_t rIdx = idx * 2 + 2;
    lIdx = lIdx >= heap.size() ? 0 : lIdx;
    rIdx = rIdx >= heap.size() ? 0 : rIdx;
    size_t sIdx = idx;
    if (lIdx && heap[sIdx] < heap[lIdx])
      sIdx = lIdx;
    if (rIdx && heap[sIdx] < heap[rIdx])
      sIdx = rIdx;
    if (idx == sIdx) return;
    swap(idx, sIdx);
    adjustDownward(sIdx);
  }

  void insertWithoutReplace(real x) {
    heap.push_back(x);
    idxs.push_back(heap.size() - 1);
    invertIdxs.push_back(heap.size() - 1);
    adjustUpward(heap.size() - 1);
  }

  void insertWithReplace(real x) {
    size_t oIdx = idxs[0];
    heap[oIdx] = heap[0] + 1;
    adjustUpward(oIdx);
    idxs.erase(idxs.begin());
    idxs.push_back(0);
    ++N;
    invertIdxs[0] = heap.size() - 1 + N;
    heap[0] = x;
    adjustDownward(0);
  }

  void resDownward(std::vector<size_t> &lHeap, size_t idx) {
    size_t lIdx = idx * 2 + 1;
    size_t rIdx = idx * 2 + 2;
    lIdx = lIdx >= lHeap.size() ? 0 : lIdx;
    rIdx = rIdx >= lHeap.size() ? 0 : rIdx;
    size_t sIdx = idx;
    if (lIdx && heap[lHeap[sIdx]] < heap[lHeap[lIdx]])
      sIdx = lIdx;
    if (rIdx && heap[lHeap[sIdx]] < heap[lHeap[rIdx]])
      sIdx = rIdx;
    if (sIdx == idx) return;
    size_t tmp = lHeap[idx];
    lHeap[idx] = lHeap[sIdx];
    lHeap[sIdx] = tmp;
    resDownward(lHeap, sIdx);
  }

  void resUpward(std::vector<size_t> &lHeap, size_t idx) {
    if (!idx) return;
    size_t pIdx = (idx - 1) / 2;
    if (heap[lHeap[idx]] > heap[lHeap[pIdx]]) {
      size_t tmp = lHeap[idx];
      lHeap[idx] = lHeap[pIdx];
      lHeap[pIdx] = tmp;
      resUpward(lHeap, pIdx);
    }
  }

  std::vector<size_t> getTopKRes(size_t K) {
    std::vector<size_t> lRes;
    std::vector<size_t> lHeap;
    lHeap.push_back(0);
    while (lRes.size() < K && lHeap.size()) {
      size_t idx = lHeap[0];
      lRes.push_back(invertIdxs[idx] - N);
      size_t lIdx = idx * 2 + 1;
      size_t rIdx = idx * 2 + 2;
      if (lIdx < heap.size()) {
        lHeap[0] = lIdx;
        resDownward(lHeap, 0);
      } else {
        lHeap[0] = lHeap.back();
        lHeap.pop_back();
        resDownward(lHeap, 0);
      }
      if (rIdx < heap.size()) {
        lHeap.push_back(rIdx);
        resUpward(lHeap, lHeap.size() - 1);
      }
    }
    return lRes;
  }

  void updateResDownward(size_t idx) {
    size_t sIdx = idx;
    size_t lIdx = idx * 2 + 1;
    size_t rIdx = idx * 2 + 2;
    lIdx = lIdx >= res.size() ? 0 : lIdx;
    rIdx = rIdx >= res.size() ? 0 : rIdx;
    if (lIdx && res[sIdx] > res[lIdx])
      sIdx = lIdx;
    if (rIdx && res[sIdx] > res[rIdx])
      sIdx = rIdx;
    if (idx == sIdx) return;
    real rTmp = res[idx];
    res[idx] = res[sIdx];
    res[sIdx] = rTmp;
    size_t bTmp = starts[idx];
    starts[idx] = starts[sIdx];
    starts[sIdx] = bTmp;
    size_t eTmp = ends[idx];
    ends[idx] = ends[sIdx];
    ends[sIdx] = eTmp;
    updateResDownward(sIdx);
  }

  void updateResUpward(size_t idx) {
    if (!idx) return;
    size_t pIdx = (idx - 1) / 2;
    if (res[pIdx] > res[idx]) {
      real rTmp = res[idx];
      res[idx] = res[pIdx];
      res[pIdx] = rTmp;
      size_t bTmp = starts[idx];
      starts[idx] = starts[pIdx];
      starts[pIdx] = bTmp;
      size_t eTmp = ends[idx];
      ends[idx] = ends[pIdx];
      ends[pIdx] = eTmp;
      updateResUpward(pIdx);
    }
  }

  void timeTraverse(real b, real e, int tIdx) {
    if (heap.size() < L)
      insertWithoutReplace(b);
    else
      insertWithReplace(b);
    std::vector<size_t> lRes = getTopKRes(K);
    size_t i = 0;
    while (i < lRes.size() && res.size() < K) {
      res.push_back(heap[idxs[lRes[i]]] * e);
      starts.push_back(tIdx - heap.size() + lRes[i] + 1);
      ends.push_back(tIdx);
      updateResUpward(res.size() - 1);
      ++i;
    }
    while (i < lRes.size() && heap[idxs[lRes[i]]] * e > res[0]) {
      res[0] = heap[idxs[lRes[i]]] * e;
      starts[0] = tIdx - heap.size() + lRes[i] + 1;
      ends[0] = tIdx;
      updateResDownward(0);
      ++i;
    }
  }

  void completeEpisode(real *bs, real *es, int seqLen) {
    for (int i = 0; i < seqLen; ++i)
      timeTraverse(bs[i], es[i], i);
  }
};

class MaxIdSequencePlusLayer : public Layer {
public:
  explicit MaxIdSequencePlusLayer(const LayerConfig& config) : Layer(config) {}

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
        second.sequenceStartPositions->getVector(false);
    size_t numSequences = first.getNumSequences();
    CHECK_EQ(numSequences, second.getNumSequences());
    const int* firstStarts = firstPositions->getData();
    const int* secondStarts = secondPositions->getData();

    real *firstVal = first.value->getData();
    real *secondVal = second.value->getData();

    IVector::resizeOrCreate(output_.ids, 2 * numSequences * config_.top_k(),
                            false);
    int *idsValue = output_.ids->getData();

    output_.value = nullptr;
    for (size_t i = 0; i < numSequences; ++i) {
      int seqLen = firstStarts[i + 1] - firstStarts[i];
      CHECK_EQ(seqLen, secondStarts[i + 1] - secondStarts[i]);
      Sequence s(config_.interval_width(), config_.top_k());
      s.completeEpisode(firstVal + firstStarts[i],
                        secondVal + secondStarts[i], seqLen);
      for (size_t j = 0; j < config_.top_k(); ++j) {
        idsValue[(i * config_.top_k() + j) * 2] = s.getStarts(j);
        idsValue[(i * config_.top_k() + j) * 2 + 1] = s.getEnds(j);
      }
    }
  }

  virtual void backward(const UpdateCallback& callback) {}
};

REGISTER_LAYER(maxid_seq_plus, MaxIdSequencePlusLayer);

}  // namespace paddle
