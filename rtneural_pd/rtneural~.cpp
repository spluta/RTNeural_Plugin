#include "m_pd.h"
//#include <memory>
#include "../RTN_Processor.cpp"
//#include <vector>

static t_class *rtneural_tilde_class;

class RTNeural_tilde {
public:
  RTNeural_tilde(t_int n_in_chans, t_int n_out_chans, t_float nn_sample_rate, t_int trig_mode);
  t_object x;

	t_canvas *canvas; // necessary for relative paths

  t_float f;
  t_float freq;
  t_float sample_rate;
  t_int blocksize;
  t_float control_rate;
  t_float nn_sample_rate;
  t_int trig_mode;
  t_int bypass;
  t_int n_in_chans;
  t_int n_out_chans;

  t_inlet *x_in2;
  t_inlet *x_in3;
  t_outlet *signal_out;

  t_float ratio;
  t_int model_loaded;

  t_int input_model_ratio;

  std::vector<t_sample*> in_vec;

  RTN_Processor processor;

  std::vector<t_float> interleaved_array;
  std::vector<t_float> outbuf;

  std::vector<t_float> in_rs;
  std::vector<t_float> out_temp;

  //for triggered input only
  std::vector<t_float> input_to_nn;
  std::vector<t_float> output_from_nn;

};  

RTNeural_tilde::RTNeural_tilde(t_int n_in_chans_a, t_int n_out_chans_a, t_float nn_sample_rate_a, t_int trig_mode_a) {
  if(n_in_chans_a<1){
    n_in_chans_a = 1;
  }
  if(n_out_chans_a<1){
    n_out_chans_a = 1;
  }

  canvas = canvas_getcurrent();

  n_in_chans = n_in_chans_a;
  n_out_chans = n_out_chans_a;
  nn_sample_rate = nn_sample_rate_a;
  
  trig_mode = trig_mode_a;
  if (trig_mode!=1) {
    trig_mode = 0;
  }
  
  x_in2 = inlet_new(&x, &x.ob_pd, &s_signal, &s_signal);
  x_in3 = inlet_new(&x, &x.ob_pd, &s_signal, &s_signal);
  signal_out = outlet_new(&x, &s_signal);

  bypass = 0;
  model_loaded = 0;

  input_model_ratio = 1;

  processor.initialize(n_in_chans, n_out_chans, 1.0f); // initialize with a dummy ratio

  //force the processor to find the local sample rate when the dsp is started
  sample_rate = 0.f;
  blocksize = 0.f;
  processor.do_resample = false;

  input_to_nn = std::vector<t_float>(n_in_chans, 0.f);
  output_from_nn = std::vector<t_float>(n_out_chans, 0.f);
  in_vec = std::vector<t_sample*>(n_in_chans);
}

static void* rtneural_tilde_new(t_floatarg n_in_chans, t_floatarg n_out_chans, t_floatarg nn_sample_rate, t_floatarg trig_mode) { 
  //first make the pd object
  RTNeural_tilde *x = (RTNeural_tilde *)pd_new(rtneural_tilde_class);
  //then run the cpp constructor
  new (x) RTNeural_tilde((t_int)n_in_chans, (t_int)n_out_chans, (t_float)nn_sample_rate, (t_int)trig_mode);
 
  return (void*)x;
}

static void RTN_bang(RTNeural_tilde *x) {
  (void)x; // silence unused variable warning
  post("stop that!");
}  


static void do_load (RTNeural_tilde *x, t_symbol* s){

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  t_int test = x->processor.load_model(absolute_path, 1);
  if(test==1){

    post("model input size: %i", x->processor.m_model_input_size);
    post("model output size: %i", x->processor.m_model_output_size);
    x->model_loaded = 1;
  } else {
    x->model_loaded = 0;
    switch(test){
      case 0:
        post("error loading the model");
        break;
      case 2:
        post("error: model input size does not match the number of input channels");
        break;
      case 3:
        post("error: model output size does not match the number of output channels");
        break;
      default:
        post("error: the path does not exist or is not a file");
        post(absolute_path);
        break;
      }
      post("disabling model");
    }
}

void rtneural_tilde_load_model(RTNeural_tilde *x, t_symbol* s){
  (void)x;

  x->model_loaded = 0;
  do_load(x, s);
}  

void rtneural_tilde_bypass(RTNeural_tilde *x, t_floatarg f){
  x->bypass = t_int(f);

  post(f ? "Bypass ON" : "Bypass OFF");
}  

void reset_vars_and_mem(RTNeural_tilde *x) {
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  t_int rs_size = t_int(ceil(x->nn_sample_rate/x->control_rate));

  //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
  t_int in_size = x->blocksize*x->n_in_chans;
  t_int in_rs_size = rs_size*x->n_in_chans;
  t_int out_temp_size = rs_size*x->n_out_chans; 
  t_int out_buf_size = x->blocksize*x->n_out_chans; 

  x->interleaved_array.resize(in_size);
  x->in_rs.resize(in_rs_size);
  x->out_temp.resize(out_temp_size);
  x->outbuf.resize(out_buf_size);

  if(x->nn_sample_rate<=0.f){
    x->ratio = 1.f;
    x->processor.do_resample = false;
  } else {
    x->ratio = x->nn_sample_rate/x->sample_rate;
    x->processor.do_resample = true;
  }

  x->processor.reset_ratio(x->ratio);

  post("ratio: %f", x->ratio);
  post("resample: %i", x->processor.do_resample);
}

void rtneural_tilde_free (RTNeural_tilde* x) {
  x->~RTNeural_tilde();
}

void rtneural_tilde_trigger_mode (RTNeural_tilde *x, t_floatarg f){
  x->trig_mode = t_int(f);
  post(f ? "Trigger mode ON" : "Trigger mode OFF");
}


t_int* rtneural_tilde_perform (t_int* args) {
  RTNeural_tilde* x = (RTNeural_tilde*)args[1];
  t_sample    *in =      (t_sample *)(args[2]);
  t_sample    *trigger =     (t_sample *)(args[3]);
  t_sample    *reset =     (t_sample *)(args[4]);
  t_sample* out = (t_sample *)args[5];
  t_int numins = (t_int)args[6];
  t_int n_samps = (t_int)args[7];

  if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
    for (t_int i = 0; i < n_samps; ++i) {
      out[i] = in[i];
    }
  } else {
    if(x->trig_mode==0){
        x->in_vec[0] = in;
        for (t_int j = 1; j < x->n_in_chans; j++) {
          x->in_vec[j] = in + j * static_cast<t_int>(n_samps);
        }

        t_int n_samps_out = x->processor.process(x->in_vec, x->input_to_nn.data(), x->in_rs.data(), x->interleaved_array.data(), x->out_temp.data(), x->outbuf.data(), n_samps);


        for (t_int j = 0; j < x->n_out_chans; j++) {
          for(t_int i = 0; i < n_samps_out; i++) {
            out[j*n_samps+ i] = (t_sample)x->outbuf[i*x->n_out_chans+j];
          }
        }
        for (t_int j = 0; j < x->n_out_chans; j++) {
          x->output_from_nn[j] = (t_sample)out[j*n_samps+n_samps-1];
        }
      //}
    } else {
        //trigger mode

        x->input_model_ratio = (numins)/x->processor.m_model_input_size;
        if(x->input_model_ratio<1){
          x->input_model_ratio = 1;
        }
        

        for (t_int i = 0; i < n_samps; ++i){
          //if reset is greater than 0, reset the model
          if(reset[i]>0.){
            x->processor.reset();
          }
          //if the trigger is greater than 0, process the input
          
          if(trigger[i]>0.){
            //the input vector could be l times larger than the model input size
            //so we need process l sets of j samples at a time
            for (int l = 0; l < x->input_model_ratio; l++) {
              for (t_int j = 0; j < x->n_in_chans; ++j) {
                x->input_to_nn[j] = (t_float)in[(j+l*x->n_in_chans)*n_samps+i];
              }
              x->processor.process1(x->input_to_nn.data(), x->output_from_nn.data());
            }
            for (t_int j = 0; j < x->n_out_chans; ++j) {
              out[j*n_samps+i] = t_sample(x->output_from_nn[j]);
            }
          } else {
            for (t_int j = 0; j < x->n_out_chans; ++j) {
              out[j*n_samps+i] = t_sample(x->output_from_nn[j]);
            }
          }
      
      }
    }
  }

  return (t_int *) (args + 8);
}

void rtneural_tilde_dsp (RTNeural_tilde* x, t_signal** sp) {
  signal_setmultiout(&sp[3], x->n_out_chans);
  if(sp[3]->s_sr != x->sample_rate || sp[3]->s_n != x->blocksize){
    x->sample_rate = sp[3]->s_sr;
    x->blocksize = sp[3]->s_n;
    reset_vars_and_mem(x);
  }

  dsp_add(rtneural_tilde_perform, 7, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_nchans, sp[0]->s_length);
}

extern "C" void rtneural_tilde_setup(void) {
  rtneural_tilde_class = class_new(gensym("rtneural~"),
    (t_newmethod)rtneural_tilde_new,
    (t_method)rtneural_tilde_free,
    sizeof(RTNeural_tilde),
    CLASS_MULTICHANNEL, A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
  
  class_addbang(rtneural_tilde_class, RTN_bang);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_load_model, gensym("load_model"), A_DEFSYMBOL, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_trigger_mode, gensym("trigger_mode"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_bypass, gensym("bypass"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(rtneural_tilde_class, RTNeural_tilde, f);
}
