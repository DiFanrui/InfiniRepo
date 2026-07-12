#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "glog/logging.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train::test {
namespace {

constexpr uint64_t kSeed = 20260614;
constexpr int64_t kWidth = 32;
constexpr int kTotalSteps = 12;
constexpr int kCheckpointStep = 5;
constexpr float kLearningRate = 0.025f;

class TinyRandomTrainingModel final : public nn::Module {
public:
    explicit TinyRandomTrainingModel(const Generator &generator)
        : nn::Module("TinyRandomTrainingModel") {
        auto weight = std::make_shared<Tensor>(
            std::vector<int64_t>{kWidth}, DataType::kFLOAT32,
            Device(Device::DeviceType::kCPU, 0), true);
        nn::init::Uniform(weight, -0.25f, 0.25f, generator);
        parameters_["weight"] = std::move(weight);
    }

    std::vector<std::shared_ptr<Tensor>>
    Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        CHECK_EQ(inputs.size(), 1);
        return {parameters_.at("weight")->Mul(inputs[0])};
    }

    void LoadWeight(const Tensor &weight) {
        parameters_.at("weight")->CopyFrom(weight);
    }

    const std::shared_ptr<Tensor> &Weight() const {
        return parameters_.at("weight");
    }
};

struct TrainingState {
    Generator generator;
    std::shared_ptr<TinyRandomTrainingModel> model;
    std::shared_ptr<optimizers::SGD> optimizer;
};

struct TrainingCheckpoint {
    int step = 0;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> generator_state;
};

TrainingState CreateTrainingState(uint64_t seed) {
    Generator generator = core::cpu::createCPUGenerator(seed);
    auto model = std::make_shared<TinyRandomTrainingModel>(generator);
    auto optimizer = std::make_shared<optimizers::SGD>(model->Parameters(), kLearningRate);
    return {std::move(generator), std::move(model), std::move(optimizer)};
}

float TrainOneStep(TrainingState &state) {
    auto input = nn::function::Rand(
        {kWidth}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0),
        state.generator, false);
    auto target = std::make_shared<Tensor>(
        std::vector<int64_t>{kWidth}, DataType::kFLOAT32,
        Device(Device::DeviceType::kCPU, 0));
    target->Fill(0.125f);

    state.optimizer->ZeroGrad();
    auto prediction = state.model->Forward({input})[0];
    auto error = prediction->Sub(target);
    auto loss = error->Pow(2.0f)->Mean(0);
    loss->Backward();
    state.optimizer->Step();

    return static_cast<const float *>(loss->DataPtr())[0];
}

std::vector<float> TrainSteps(TrainingState &state, int count) {
    std::vector<float> losses;
    losses.reserve(count);
    for (int step = 0; step < count; ++step) {
        losses.push_back(TrainOneStep(state));
    }
    return losses;
}

TrainingCheckpoint SaveCheckpoint(const TrainingState &state, int step) {
    const auto state_dict = state.model->StateDict();
    const auto &source_weight = state_dict.at("weight");
    auto weight = std::make_shared<Tensor>(
        source_weight->Dims(), source_weight->Dtype(),
        Device(Device::DeviceType::kCPU, 0), true);
    weight->CopyFrom(source_weight);
    return {step, std::move(weight), state.generator.get_state()};
}

TrainingState RestoreCheckpoint(const TrainingCheckpoint &checkpoint,
                                bool restore_generator_state) {
    // Recreate the stateless SGD optimizer from the same hyperparameters.
    // When optimizer state serialization is added to InfiniTrain, this is the
    // place to restore momentum/Adam buffers as well.
    TrainingState state = CreateTrainingState(kSeed);
    state.model->LoadWeight(*checkpoint.weight);
    if (restore_generator_state) {
        state.generator.set_state(*checkpoint.generator_state);
    }
    return state;
}

std::vector<float> TensorValues(const std::shared_ptr<Tensor> &tensor) {
    const auto *data = static_cast<const float *>(tensor->DataPtr());
    return std::vector<float>(data, data + tensor->NumElements());
}

TEST(GeneratorTrainingResumeTest, RestoredRngReproducesUninterruptedTraining) {
    TrainingState uninterrupted = CreateTrainingState(kSeed);
    const std::vector<float> uninterrupted_losses = TrainSteps(uninterrupted, kTotalSteps);

    TrainingState first_part = CreateTrainingState(kSeed);
    TrainSteps(first_part, kCheckpointStep);
    const TrainingCheckpoint checkpoint = SaveCheckpoint(first_part, kCheckpointStep);
    ASSERT_EQ(checkpoint.step, kCheckpointStep);

    TrainingState resumed = RestoreCheckpoint(checkpoint, true);
    const std::vector<float> resumed_losses
        = TrainSteps(resumed, kTotalSteps - checkpoint.step);

    const std::vector<float> expected_suffix(
        uninterrupted_losses.begin() + checkpoint.step,
        uninterrupted_losses.end());
    EXPECT_EQ(resumed_losses, expected_suffix);
    EXPECT_EQ(TensorValues(resumed.model->Weight()),
              TensorValues(uninterrupted.model->Weight()));
}

TEST(GeneratorTrainingResumeTest, OmittingRngStateChangesTrainingTrajectory) {
    TrainingState first_part = CreateTrainingState(kSeed);
    TrainSteps(first_part, kCheckpointStep);
    const TrainingCheckpoint checkpoint = SaveCheckpoint(first_part, kCheckpointStep);

    TrainingState restored = RestoreCheckpoint(checkpoint, true);
    TrainingState rng_not_restored = RestoreCheckpoint(checkpoint, false);
    const std::vector<float> restored_losses
        = TrainSteps(restored, kTotalSteps - checkpoint.step);
    const std::vector<float> divergent_losses
        = TrainSteps(rng_not_restored, kTotalSteps - checkpoint.step);

    EXPECT_NE(divergent_losses, restored_losses);
    EXPECT_NE(TensorValues(rng_not_restored.model->Weight()),
              TensorValues(restored.model->Weight()));
}

} // namespace
} // namespace infini_train::test
