#include "RTNeuralUGen.hpp"
#include "SC_PlugIn.hpp"
#include "SC_PlugIn.h"
#include <string>


static InterfaceTable *ft;


  RTNeuralUGen::RTNeuralUGen()
  {
    m_num_input_chans = numInputs()-4;
    m_num_output_chans = numOutputs();

    int nn_sample_rate = in0(1);

    //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
    int in_size = bufferSize()*m_num_input_chans;
    int in_rs_size = int(ceil(nn_sample_rate/controlRate())*m_num_input_chans);
    int out_temp_size = int(ceil(nn_sample_rate/controlRate())*m_num_output_chans); 
    int out_buf_size = bufferSize()*m_num_output_chans; 

    interleaved_array = (float*)RTAlloc(mWorld, (double)in_size * sizeof(float));
    in_rs = (float*)RTAlloc(mWorld, (double)in_rs_size * sizeof(float));
    out_temp = (float*)RTAlloc(mWorld, (double)out_temp_size * sizeof(float));
    outbuf = (float*)RTAlloc(mWorld, (double)out_buf_size * sizeof(float));

    input_to_nn = (float*)RTAlloc(mWorld, (double)m_num_input_chans*sizeof(float));
    output_from_nn = (float*)RTAlloc(mWorld, (double)m_num_output_chans*sizeof(float));

    ins = (float const**)RTAlloc(mWorld, (double)m_num_input_chans*sizeof(float*));
    outs = (float **)RTAlloc(mWorld, (double)m_num_output_chans*sizeof(float*));

    float ratio = 1.f;
    //if the model was trained at a different sample rate, we need to resample the input and output
    if(nn_sample_rate>0.f) {
      ratio = (float)nn_sample_rate / (float)sampleRate();
    }
    if(ratio!=1.f) {
      processor.do_resample = true;
    } else {
      processor.do_resample = false;
    }
    std::cout<<"do_resample: "<<processor.do_resample<<std::endl;

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

    //suspend inference while loading the model
    unit->processor.m_model_loaded=false;

    int test = unit->processor.load_model(pathStr, verbose);

    std::cout<<"model loaded: "<<unit->processor.m_model_loaded<<path<<std::endl;

    if(test==1){
      std::cout<<"model input size: " << unit->processor.m_model_input_size<<std::endl;
      std::cout<<"model output size: " << unit->processor.m_model_output_size<<std::endl;
    }
    else {
      switch(test){
        case 0:
          std::cout<<"error loading the model"<<std::endl;
          break;
        case 2:
          std::cout<<"error: model input size does not match the number of input channels"<<std::endl;
          break;
        case 3:
          std::cout<<"error: model output size does not match the number of output channels"<<std::endl;
          break;
        default:
          std::cout<<"error: the path does not exist or is not a file";
          break;
        }
        std::cout<<"disabling model"<<std::endl;
      }
}

void RTNeuralUGen::next(int nSamples)
{
  const float bypass = in0(0);
  const float trig_mode = in0(2);
  const float* trigger = in(3);
  
  for (int i = 0; i < m_num_input_chans; ++i) {
    ins[i] = in(i+4);
  }
  for (int i = 0; i < m_num_output_chans; ++i) {
    outs[i] = out(i);
  }

  if ((processor.m_model_loaded==false)||((int)bypass==1)) {
    for (int i = 0; i < nSamples; ++i) {
      int small = std::min(m_num_input_chans, m_num_output_chans);
      for (int j = 0; j < m_num_output_chans; ++j) {
        if(j<small) {
          outs[j][i] = ins[j][i];
        } else {
          outs[j][i] = 0.f;
        }
      }
    }
  } 
  else {
    if(trig_mode==0) {

      int n_samps_out = processor.process(ins, input_to_nn, in_rs, interleaved_array, out_temp, outbuf, nSamples);

      for (int j = 0; j < m_num_output_chans; j++) {
        for(int i = 0; i < nSamples; i++) {
          outs[j][i] = outbuf[i*m_num_output_chans+j];
        }
      }

    } else {
      for (int i = 0; i < nSamples; ++i){
        if(trigger[i]>0.f){
          //std::cout<<"triggered"<<std::endl;
          for (int j = 0; j < m_num_input_chans; ++j) {
            input_to_nn[j] = (float)ins[j][i];
          }
          processor.process1(input_to_nn, output_from_nn);
          for (int j = 0; j < m_num_output_chans; ++j) {
            outs[j][i] = output_from_nn[j];
          }
        } else {
          for (int j = 0; j < m_num_output_chans; ++j) {
            outs[j][i] = output_from_nn[j];
          }
        }
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
