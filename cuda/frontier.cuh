#pragma once
#include <array>

namespace nano::frontier {
// PR #360's shipped source, not its stale prose schedule or current merged master.
inline constexpr char commit[] = "c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7";
inline constexpr int model_dim = 768, mlp_width = 2816, heads = 6, layers = 11;
inline constexpr int scheduled_steps = 1174, total_steps = 1194, untie_step = 1175;
inline constexpr int ngram_rows = 84602880, ngram_dim = 768, sign_rows = 8192;
inline constexpr int validation_tokens = 10485760, vocabulary = 50304;
inline constexpr int attention_segment_cap = 2560;
inline constexpr std::array<int, 4> unique_local_tokens{16384, 32768, 49152, 40960};
// Zero means the sublayer is absent. Layer 8 also executes MLP bank slot 11.
inline constexpr std::array<int, 11> qk_width{64, 64, 64, 128, 0, 64, 0, 0, 64, 0, 128};
inline constexpr std::array<int, 11> value_width{128, 64, 128, 128, 0, 128, 0, 0, 64, 0, 128};
inline constexpr std::array<int, 11> mlp_bank_order{0, 1, 2, 3, 4, 5, 6, 8, 11, 9, 10};

struct Stage {
    int begin, end, local_tokens, sequence_limit, short_window, long_window;
    double lr_multiplier;
};
inline constexpr std::array<Stage, 5> stages{{
    {0, 320, 16384, 896, 1, 3, 1.0},
    {320, 681, 32768, 2048, 3, 7, 1.52},
    {681, 1107, 49152, 3072, 5, 11, 1.73},
    {1107, 1174, 40960, 3072, 5, 11, 1.579266707473229},
    {1174, 1194, 16384, 3072, 6, 13, 1.0},
}};
static_assert(stages.back().end == total_steps);
static_assert(ngram_rows % 8 == 0);
} // namespace nano::frontier
