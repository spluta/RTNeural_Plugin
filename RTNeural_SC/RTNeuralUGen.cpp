#include "RTNeuralUGen.hpp"
#include "SC_PlugIn.hpp"
#include "SC_PlugIn.h"
#include <string>


static InterfaceTable *ft;


  RTNeuralUGen::RTNeuralUGen()
  {
    m_num_input_chans = numInputs()-2;
    m_num_output_chans = numOutputs();

    in_vec.resize(m_num_input_chans);

    //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
    int in_size = bufferSize()*m_num_input_chans;
    int in_rs_size = int(ceil(in0(1)/controlRate())*m_num_input_chans);
    int out_temp_size = int(ceil(in0(1)/controlRate())*m_num_output_chans); 
    int out_buf_size = bufferSize()*m_num_output_chans; 

    interleaved_array = (float*)RTAlloc(mWorld, (double)in_size * sizeof(float));
    in_rs = (float*)RTAlloc(mWorld, (double)in_rs_size * sizeof(float));
    out_temp = (float*)RTAlloc(mWorld, (double)out_temp_size * sizeof(float));
    outbuf = (float*)RTAlloc(mWorld, (double)out_buf_size * sizeof(float));

    //RTN_Processor processor;

    float ratio = 1.f;
    if(in0(1)>0.f) {
      ratio = in0(1) / sampleRate();
    }

    processor.initialize(m_num_input_chans, m_num_output_chans, ratio);

    mCalcFunc = make_calc_function<RTNeuralUGen, &RTNeuralUGen::next>();
    next(1);
  }

  RTNeuralUGen::~RTNeuralUGen() {
    RTFree(mWorld, in_rs);
    RTFree(mWorld, out_temp);
    RTFree(mWorld, outbuf);
    RTFree(mWorld, interleaved_array);
  }

  void load_model (RTNeuralUGen* unit, sc_msg_iter* args) {
    const char *path = args->gets();
    const bool verbose = args->geti();

    std::string pathStr = path;

    unit->processor.load_model(pathStr, verbose);
}

void RTNeuralUGen::next(int nSamples)
{
  const float bypass = in0(0);

  if ((processor.m_model_loaded==false)||((int)bypass==1)) {
    for (int i = 0; i < nSamples; ++i) {
      out(0)[i] = in(2)[i];
    }
  } else {
    for (int j = 0; j < m_num_input_chans; ++j) {
      in_vec[j] = in(2+j);
    }

    int n_samps_out = processor.process(in_vec, in_rs, interleaved_array, out_temp, outbuf, nSamples);

    //deinterleave the output and put it in the output buffers
    for(int i = 0; i < n_samps_out; ++i) {
      for (int j = 0; j < m_num_output_chans; ++j) {
        out(j)[i] = outbuf[i*m_num_output_chans+j];
      }
    }
  }
}

PluginLoad(RTNeural)
{
  ft = inTable;
  registerUnit<RTNeuralUGen>(ft, "RTNeural", false);
  DefineUnitCmd("RTNeural", "load_model", (UnitCmdFunc)&load_model);
}
