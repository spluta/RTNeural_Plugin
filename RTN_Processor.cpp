#include "RTN_Processor.hpp"

size_t RTN_Processor::resample (const float* input, float* output, size_t numSamples) noexcept
{
    SRC_DATA src_data {
        input, // data_in
        output, // data_out
        (int) numSamples, // input_frames
        int ((double) numSamples * m_ratio) + 1, // output_frames
        0, // input_frames_used
        0, // output_frames_gen
        0, // end_of_input
        m_ratio // src_ratio
    };

    src_process (src_state.get(), &src_data);
    return (size_t) src_data.output_frames_gen;
}

size_t RTN_Processor::resample_out (const float* input, float* output, size_t inSamples, size_t outSamples) noexcept
{
    SRC_DATA src_data {
        input, // data_in
        output, // data_out
        (int) inSamples, // input_frames
        (int) outSamples, // output_frames
        0, // input_frames_used
        0, // output_frames_gen
        0, // end_of_input
        1./m_ratio // src_ratio
    };

    src_process (src_state_out.get(), &src_data);
    return (size_t) src_data.output_frames_gen;
}

RTN_Processor::RTN_Processor(){}

void RTN_Processor::reset_ratio(float ratio){
    m_ratio = ratio;
    src_set_ratio (src_state.get(), ratio);
    src_set_ratio (src_state_out.get(), 1./ratio);
} 

void RTN_Processor::initialize(int num_inputs, int num_outputs, float ratio)
{
    m_num_in_chans = num_inputs;
    m_num_out_chans = num_outputs;

    m_ratio = ratio;

    //in_vec.resize(num_inputs);
    inVecSmall.resize(num_inputs);

    std::cout << "num_inputs: " << num_inputs << " num_outputs: " << num_outputs << std::endl;

    int error;
    int error_out;
    src_state.reset (src_new (SRC_SINC_BEST_QUALITY, num_inputs, &error));
    src_state_out.reset (src_new (SRC_SINC_BEST_QUALITY, num_outputs, &error_out));
    src_set_ratio (src_state.get(), ratio);
    src_set_ratio (src_state_out.get(), 1./ratio);
}

RTN_Processor::~RTN_Processor()
{
    src_delete (src_state.release());
    src_delete (src_state_out.release());
}

int RTN_Processor::load_model(std::string pathStr, int verbose) {
    try {

      //return 4 if the path does not exist or is not a file
      if (!std::filesystem::exists(pathStr) || !std::filesystem::is_regular_file(pathStr)) {
        return 4;
      }

      std::ifstream jsonStream(pathStr, std::ifstream::binary);

      m_model = RTNeural::json_parser::parseJson<float>(jsonStream, false);

      m_model_input_size = m_model->layers[0]->in_size;
      m_model_output_size = m_model->layers[m_model->layers.size()-1]->out_size;

      if (m_model_input_size!=m_num_in_chans) {
        m_model_loaded = false;
        return 2;
      }

      if (m_model_output_size!=m_num_out_chans) {
        m_model_loaded = false;
        return 3;
      }

      m_model_loaded = true;
      m_model->reset();
      return 1;

    } catch (const std::exception& e) {
      m_model_loaded = false;
      return 0;
    }
    return 1;
}

//processes the input data through the model
//receives 5 data structures that must be declared in the parent process: 
//1) a vector of pointers to the N channels of float* (a pointer per input channel)
//2) a vector for resampling the input data
//3) a vector to store the interleaved input data
//4) a vector to return the output data
//5) the number of input samples
//the process function returns the number of output samples, which should match the number of input samples
//the output data is interleaved into the 'outbuf' and must be deinterleaved in the parent process
//uses a template so in_vec can ba a vector of pointers to floats or a multi-dimensional array
template <typename T>
int RTN_Processor::process(T in_vec, float* in_rs, float* interleaved_array, float* out_temp, float* outbuf, int nSamples) {
  if (m_ratio == 1.0f) {
    for (int i = 0; i < nSamples; ++i) {
      for (int j = 0; j < m_num_in_chans; ++j) {
        inVecSmall[j] = static_cast<float>(in_vec[j][i]);
      }
      
      process1(inVecSmall.data(), outbuf + i * m_num_out_chans);
    }
    return nSamples;
  } else {
    // if the model relies on a sample rate that is different than the current rate,
    // we need to interleave and resample the input data
    for (int i = 0; i < nSamples; ++i) {
      for (int j = 0; j < m_num_in_chans; ++j) {
        interleaved_array[i * m_num_in_chans + j] = static_cast<float>(in_vec[j][i]);
      }
    }

    // resample the input to the model's sample rate
    int resampled_size = resample(interleaved_array, in_rs, nSamples);

    // run the model on the resampled audio
    for (int i = 0; i < resampled_size; ++i) {
      for (int j = 0; j < m_num_in_chans; ++j) {
        inVecSmall[j] = in_rs[i * m_num_in_chans + j];
      }
      // out_temp[i * m_num_out_chans] = m_model->forward(inVecSmall.data());
      // if (m_num_out_chans > 1) {
      //   auto vec = m_model->getOutputs();
      //   for (int j = 1; j < m_num_out_chans; j++) {
      //     out_temp[i * m_num_out_chans + j] = vec[j];
      //   }
      // }
      process1(inVecSmall.data(), out_temp + i * m_num_out_chans);
    }

    // resample the output back to the original sample rate
    int n_samps_out = resample_out(out_temp, outbuf, resampled_size, nSamples);
    return n_samps_out;
  }
}

void RTN_Processor::process1(const float* input, float* output) {
  output[0] = m_model->forward(input);
  if(m_num_out_chans>1) {
    auto vec = m_model->getOutputs();
    for (int j = 1; j < m_num_out_chans; j++) {
      output[j] = vec[j];
    }
  }
}