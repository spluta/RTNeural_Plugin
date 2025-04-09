#include "m_pd.h"
//#include <memory>
#include "../RTN_Processor.cpp"
#include <thread>
//#include <vector>

static t_class *rtneural_tilde_class;

typedef struct _rtneural_tilde {
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

	t_outlet *signal_out;

  t_float ratio;
  t_int model_loaded;

  t_sample** in_vec;

  RTN_Processor processor;

  float* interleaved_array;
  float* outbuf;

  float *in_rs;
  float *out_temp;

  //for triggered input only
  float* input_to_nn;
  float* output_from_nn;

  std::thread loading_thread;
  std::atomic<bool> thread_running{false};

} t_rtneural_tilde;  

void rtneural_tilde_bang(t_rtneural_tilde *x) {
  (void)x; // silence unused variable warning
  post("stop that!");
}  

void do_load (t_rtneural_tilde *x, t_symbol* s){

  post("loading model: ");
  post(s->s_name);

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
        break;
      }
      post("disabling model");
    }
}

void rtneural_tilde_load_model(t_rtneural_tilde *x, t_symbol* s){
  (void)x;

  x->model_loaded = 0;
  do_load(x, s);
}  

void rtneural_tilde_bypass(t_rtneural_tilde *x, t_floatarg f){
  x->bypass = t_int(f);

  post(f ? "Bypass ON" : "Bypass OFF");
}  

void reset_vars_and_mem(t_rtneural_tilde *x) {
  x->sample_rate = sys_getsr();
  post("sample rate: %f", x->sample_rate);
  x->blocksize = sys_getblksize();
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  t_int rs_size = t_int(ceil(x->nn_sample_rate/x->control_rate));

  //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
  t_int in_size = x->blocksize*x->n_in_chans;
  t_int in_rs_size = rs_size*x->n_in_chans;
  t_int out_temp_size = rs_size*x->n_out_chans; 
  t_int out_buf_size = x->blocksize*x->n_out_chans; 

  x->interleaved_array = (t_float*)calloc(in_size, sizeof(t_float));
  x->in_rs = (t_float*)calloc(in_rs_size, sizeof(t_float));
  x->out_temp = (t_float*)calloc(out_temp_size, sizeof(t_float));
  x->outbuf = (t_float*)calloc(out_buf_size, sizeof(t_float));
  x->in_vec = (t_sample**)calloc(x->n_in_chans, sizeof(t_sample*));

  if(x->nn_sample_rate<=0.f){
    x->ratio = 1.f;
  } else {
    x->ratio = x->nn_sample_rate/x->sample_rate;
  }
}

void* rtneural_tilde_new(t_floatarg n_in_chans, t_floatarg n_out_chans, t_floatarg nn_sample_rate, t_floatarg trig_mode) { 
  
  if(n_in_chans<1.f){
    n_in_chans = 1.f;
  }
  if(n_out_chans<1.f){
    n_out_chans = 1.f;
  }
  t_rtneural_tilde *x = (t_rtneural_tilde *)pd_new(rtneural_tilde_class);
  x->canvas = canvas_getcurrent();

  x->n_in_chans = t_int(n_in_chans);
  x->n_out_chans = t_int(n_out_chans);
  x->nn_sample_rate = nn_sample_rate;
  
  if (trig_mode!=1.f) {
    trig_mode = 0.f;
  }
  x->trig_mode = t_int(trig_mode);

  x->signal_out = outlet_new(&x->x, &s_signal);

  x->bypass = 0;
  x->model_loaded = 0;

  reset_vars_and_mem(x);

  x->processor.initialize(x->n_in_chans, x->n_out_chans, x->ratio);

  if(x->ratio==1.f){
    x->processor.do_resample = false;
  } else {
    x->processor.do_resample = true;
  }

  post("ratio: %f", x->ratio);
  post("resample: %i", x->processor.do_resample);

  x->input_to_nn = (t_sample*)calloc(x->n_in_chans, sizeof(t_sample));
  x->output_from_nn = (t_sample*)calloc(x->n_out_chans, sizeof(t_sample));

  return (void *)x;
}

void rtneural_tilde_free (t_rtneural_tilde* x) {
  free(x->interleaved_array);
  free(x->in_rs);
  free(x->out_temp);
  free(x->outbuf);
  free(x->input_to_nn);
  free(x->output_from_nn);
  free(x->in_vec);

  x->processor.~RTN_Processor();

	outlet_free(x->signal_out);
}

void rtneural_tilde_trigger_mode (t_rtneural_tilde *x, t_floatarg f){
  x->trig_mode = t_int(f);
  post(f ? "Trigger mode ON" : "Trigger mode OFF");
}

t_int* rtneural_tilde_perform (t_int* args) {
  t_rtneural_tilde* x = (t_rtneural_tilde*)args[1];
  t_sample    *in =      (t_sample *)(args[2]);
  t_sample* out = (t_sample *)args[3];
  t_int n_samps = (t_int)args[4];


  if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
    for (t_int i = 0; i < x->blocksize; ++i) {
      out[i] = in[i];
    }
  } else {
    if(x->trig_mode==0){
      // //this should not be neeeded, but pd crashes the other way - I don't know why
      // if (x->processor.do_resample==false) {
      //   for (t_int i = 0; i < x->blocksize; ++i){
      //     for (t_int j = 0; j < x->n_in_chans; ++j) {
      //       x->input_to_nn[j] = (float)in[j*x->blocksize+i];
      //     }
      //     x->processor.process1(x->input_to_nn, x->output_from_nn);
      //     for (t_int j = 0; j < x->n_out_chans; ++j) {
      //       out[j*x->blocksize+i] = t_sample(x->output_from_nn[j]);
      //     }
      //   }
      // } else {
        //only run this if the ratio is not 1
        x->in_vec[0] = in;
        for (t_int j = 1; j < x->n_in_chans; j++) {
          x->in_vec[j] = in + j * static_cast<t_int>(x->blocksize);
        }

        t_int n_samps_out = x->processor.process(x->in_vec, x->input_to_nn, x->in_rs, x->interleaved_array, x->out_temp, x->outbuf, x->blocksize);

        //post("n_samps_out: %i", n_samps_out);

        // for (t_int j = 0; j < x->n_out_chans; j++) {
        //   for(t_int i = 0; i < x->blocksize; i++) {
        //     outs[j][i] = (double)x->outbuf[i*x->n_out_chans+j];
        //   }
        // }

        for (t_int j = 0; j < x->n_out_chans; j++) {
          for(t_int i = 0; i < n_samps_out; i++) {
            out[j*x->blocksize+ i] = (t_sample)x->outbuf[i*x->n_out_chans+j];
          }
        }
      //}
    } else {
      if(n_samps>x->n_in_chans*x->blocksize){
        for (t_int i = 0; i < x->blocksize; ++i){
          if(in[x->n_in_chans*x->blocksize+i]>0.){
            for (t_int j = 0; j < x->n_in_chans; ++j) {
              x->input_to_nn[j] = (float)in[j*x->blocksize+i];
            }
            x->processor.process1(x->input_to_nn, x->output_from_nn);
            for (t_int j = 0; j < x->n_out_chans; ++j) {
              out[j*x->blocksize+i] = t_sample(x->output_from_nn[j]);
            }
          } else {
            for (t_int j = 0; j < x->n_out_chans; ++j) {
              out[j*x->blocksize+i] = t_sample(x->output_from_nn[j]);
            }
          }
        }
      }
    }
  }

  return (t_int *) (args + 5);
}

void rtneural_tilde_dsp (t_rtneural_tilde* x, t_signal** sp) {
  signal_setmultiout(&sp[1], x->n_out_chans);
  if(sys_getsr() != x->sample_rate || sys_getblksize()!=x->blocksize){
    reset_vars_and_mem(x);
    x->processor.reset_ratio(x->ratio);
  }

  dsp_add(rtneural_tilde_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, (t_int)(sp[0]->s_length * sp[0]->s_nchans));
}

#if defined(_LANGUAGE_C_PLUS_PLUS) || defined(__cplusplus)
extern "C" {
  void rtneural_tilde_setup(void);
}
#endif

void rtneural_tilde_setup(void) {
  rtneural_tilde_class = class_new(gensym("rtneural~"),
    (t_newmethod)rtneural_tilde_new,
    (t_method)rtneural_tilde_free,
    sizeof(t_rtneural_tilde),
    CLASS_MULTICHANNEL, A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
  
  class_addbang(rtneural_tilde_class, rtneural_tilde_bang);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_load_model, gensym("load_model"), A_DEFSYMBOL, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_trigger_mode, gensym("trigger_mode"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_bypass, gensym("bypass"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_tilde_class, (t_method)rtneural_tilde_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(rtneural_tilde_class, t_rtneural_tilde, f);
}
