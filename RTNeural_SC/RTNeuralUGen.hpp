#include "SC_PlugIn.hpp"
#include "../RTN_Processor.cpp"

class RTNeuralUGen : public SCUnit {
public:
  RTNeuralUGen();

  // Destructor
  ~RTNeuralUGen();

  RTN_Processor processor;

  int m_num_input_chans;
  int m_num_output_chans;

  float* input_to_nn;
  float* output_from_nn;

  float* interleaved_array;
  float* outbuf;

  float const** ins;
  float** outs;

private:
  // Calc function
  void next(int nSamples);

  void load_model(RTNeuralUGen* unit, sc_msg_iter* args);

  float *in_rs;
  float *out_temp;

  int m_bypass{1};
};


