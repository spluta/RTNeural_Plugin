//#include <RTNeural/RTNeural.h>
#include "RTNeuralCPP/RTNeural/RTNeural.h"
#include "libsamplerate/include/samplerate.h"

class RTN_Processor { 
public:

    RTN_Processor();
    ~RTN_Processor();

    size_t resample (const float* input, float* output, size_t numSamples) noexcept;
    size_t resample_out (const float* input, float* output, size_t inSamples, size_t outSamples) noexcept;

    std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state { nullptr, &src_delete };
    std::unique_ptr<SRC_STATE, decltype (&src_delete)> src_state_out { nullptr, &src_delete };

    bool m_model_loaded{false};
    std::unique_ptr<RTNeural::Model<float>> m_model;

    int m_model_input_size{1};
    int m_model_output_size{1};

    int m_num_in_chans;
    int m_num_out_chans;

    float m_ratio{1.0f};
    bool do_resample{false};

    void reset_ratio(float ratio);
    void initialize(int num_inputs, int num_outputs, float sr_in);

    int load_model(std::string pathStr, int verbose);
    void reset();

    //template <typename T> T process(T in_vec, float* in_rs, float* interleaved_array, float* out_temp, float* outbuf, int nSamples);
    // int process(const std::vector<const float*>& in, float* in_rs, float* interleaved_array, float* out_temp, float* outbuf, int nSamples);
    template <typename T, typename U>
    int process(T in_vec, U in_vec_small, U in_rs, U interleaved_array, U out_temp, U outbuf, int nSamples);

    template <typename T, typename U>
    void process1(T input, U output);
};