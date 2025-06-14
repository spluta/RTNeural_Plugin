/**
	@file
	rtneural~ - run real-time audio-rate neural networks in Max/MSP

	@ingroup	examples

	Copyright 2025 - Sam Pluta
*/

#include "ext.h"
#include "ext_obex.h"
#include "commonsyms.h"
#include "z_dsp.h"

#include "../../RTN_Processor.cpp"

#include <vector>
using namespace std;

// max xect instance data
class t_rtneural_tilde {
public:

  t_rtneural_tilde(t_symbol *s, long argc, t_atom *argv);

  t_pxobject m_obj;
  
  float sample_rate;
  t_int blocksize;
  float control_rate;
  t_int bypass;
  t_int n_in_chans;
  t_int n_out_chans;
  float nn_sample_rate;
  long trig_mode;

  float ratio;
  float model_loaded;

  t_int input_model_ratio;

	RTN_Processor processor;

  std::vector<float> interleaved_array;
  std::vector<float> outbuf;

  std::vector<float> in_rs;
  std::vector<float> out_temp;

  //for triggered input only
  std::vector<float> input_to_nn;
  std::vector<float> output_from_nn;

}; 

// prototypes
void *rtneural_tilde_new(t_symbol *s, long argc, t_atom *argv);
void rtneural_tilde_free(t_rtneural_tilde *x);
void rtneural_tilde_load_model(t_rtneural_tilde *x, t_symbol s, long argc, t_atom *argv);
void rtneural_tilde_reset(t_rtneural_tilde *x, long f);
void rtneural_tilde_bypass(t_rtneural_tilde *x, long f);

void rtneural_tilde_perform64(t_rtneural_tilde *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void rtneural_tilde_dsp64(t_rtneural_tilde *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void reset_vars_and_mem(t_rtneural_tilde *x, float sample_rate, t_int blocksize);
long rtneural_tilde_multichanneloutputs(t_rtneural_tilde *x, long outletindex);
void rtneural_tilde_assist(t_rtneural_tilde *x, void *b, long io, long index, char *s);

// globals
static t_class *rtneural_tilde_class;

/************************************************************************************/


t_rtneural_tilde::t_rtneural_tilde(t_symbol *s, long argc, t_atom *argv) {
  post("new rtneural~ object");

  float n_in_chans_a = atom_getfloatarg(0, argc, argv);
  float n_out_chans_a = atom_getfloatarg(1, argc, argv);
  float nn_sample_rate_a = atom_getfloatarg(2, argc, argv);
  float trig_mode_a = atom_getfloatarg(3, argc, argv);

  input_model_ratio = 1;

  if(n_in_chans_a<1.f){
    n_in_chans_a = 1.f;
  }
  if(n_out_chans_a<1.f){
    n_out_chans_a = 1.f;
  }
  n_in_chans = (t_int)n_in_chans_a;
  n_out_chans = (t_int)n_out_chans_a;

  input_to_nn.resize(n_in_chans, 0.f);
  output_from_nn.resize(n_out_chans, 0.f);

  for(int i=0; i<n_in_chans; i++){
    input_to_nn[i] = 0.f;
  }
  for(int i=0; i<n_out_chans; i++){
    output_from_nn[i] = 0.f;
  }
  
  nn_sample_rate = nn_sample_rate_a;

  if (trig_mode_a!=1.f) {
    trig_mode_a = 0.f;
  }
  trig_mode = (long)trig_mode_a;

  bypass = 0;
  model_loaded = 0.f;

  //reset_vars_and_mem(x, sys_getsr(), sys_getblksize());
  sample_rate = 0.f;
  blocksize = 0.f;
  control_rate = 0.f;
  processor.do_resample = false;

  processor.initialize(n_in_chans, n_out_chans, 1.0f); // initialize with a dummy ratio
}

void *rtneural_tilde_new(t_symbol *s, long argc, t_atom *argv)
{
	t_rtneural_tilde *x = (t_rtneural_tilde *)object_alloc(rtneural_tilde_class);
  new (x) t_rtneural_tilde(s, argc, argv);

  dsp_setup((t_pxobject *)x, 3);
	x->m_obj.z_misc |= Z_NO_INPLACE | Z_MC_INLETS;
	outlet_new((t_object *)x, "multichannelsignal");
	
	return x;
}

void rtneural_tilde_free (t_rtneural_tilde* x) {
  z_dsp_free((t_pxobject *)x);
  x->~t_rtneural_tilde();
}

void ext_main(void *r)
{
  t_class	*c = class_new("rtneural~",
      (method)rtneural_tilde_new,
      (method)rtneural_tilde_free, sizeof(t_rtneural_tilde),
      (method)NULL,
      A_GIMME, 0);

  class_addmethod(c, (method)rtneural_tilde_load_model, "load_model", A_GIMME, 0);
  class_addmethod(c, (method)rtneural_tilde_reset, "reset", A_GIMME, 0);
  class_addmethod(c, (method)rtneural_tilde_bypass, "bypass", A_LONG, 0);
  class_addmethod(c, (method)rtneural_tilde_dsp64, "dsp64",	A_CANT, 0); 
  class_addmethod(c, (method)rtneural_tilde_assist, "assist", A_CANT, 0); 

  class_addmethod(c, (method)rtneural_tilde_multichanneloutputs, "multichanneloutputs", A_CANT, 0);

  CLASS_ATTR_LONG(c, "trig_mode", 0, t_rtneural_tilde, trig_mode);
  CLASS_ATTR_FILTER_CLIP(c, "trig_mode", 0, 1);

  class_dspinit(c);
	class_register(CLASS_BOX, c);
	rtneural_tilde_class = c;
}

/************************************************************************************/

long rtneural_tilde_multichanneloutputs(t_rtneural_tilde *x, long outletindex)
{
  return x->n_out_chans; 
}

void reset_vars_and_mem(t_rtneural_tilde *x, float sample_rate, t_int blocksize){
  post("resetting nn sample rate and block size B");

  x->sample_rate = sample_rate;
  x->blocksize = blocksize;
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  if(x->nn_sample_rate<=0.f){
    x->ratio = 1.f;
    x->processor.do_resample = false;
  } else {
    x->ratio = x->nn_sample_rate/x->sample_rate;
    if(x->ratio==1.f){
      x->processor.do_resample = false;
    } else {
      x->processor.do_resample = true;
    }
  }
  post("ratio: %f, resample: %i", x->ratio, x->processor.do_resample);

  t_int rs_size = t_int(ceil(x->nn_sample_rate/x->control_rate));

  t_int in_size = x->blocksize*x->n_in_chans;
  t_int in_rs_size = rs_size*x->n_in_chans;
  t_int out_temp_size = rs_size*x->n_out_chans; 
  t_int out_buf_size = x->blocksize*x->n_out_chans;

  x->sample_rate = sample_rate;
  x->blocksize = blocksize;
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  x->interleaved_array.resize(in_size, 0.f);
  x->in_rs.resize(in_rs_size, 0.f);
  x->out_temp.resize(out_temp_size, 0.f);
  x->outbuf.resize(out_buf_size, 0.f);

}

t_int get_abs_path(t_symbol *path_in, char* filename, int read_write){
  t_fourcc filetype = 'JSON', outtype;
  short numtypes = 1;
  
  short path;

  if (path_in == gensym("")) {      // if no argument supplied, ask for file
      if (read_write==1) {
          if (saveas_dialog(filename, &path, NULL)) // non-zero: user cancelled
              return 1;
      } else {
          if (open_dialog(filename, &path, &outtype, &filetype, numtypes))       // non-zero: user cancelled
              return 1;
      }
        
  } else {
      strcpy(filename, path_in->s_name);    // must copy symbol before calling locatefile_extended
      if (locatefile_extended(filename, &path, &outtype, &filetype, 1)) { // non-zero: not found
          //object_error(x, "%s: not found", path_in->s_name);
          post("Failed to open input file");
          return 1;
      }
  }

  if(path_toabsolutesystempath(path, filename, filename)){
    post("Failed to get absolute path");
    return 1;
  }

  return 0;
}

void rtneural_tilde_doload_model(t_rtneural_tilde *x, t_symbol *path_in){
  char filename[MAX_PATH_CHARS];
  if(get_abs_path(path_in, filename, 0)){
    return;
  }

  post("loading model: ");
  post(filename);

  t_int test = x->processor.load_model(filename, 1);
  if(test==1){
    x->model_loaded = 1;

    post("model input size: %i", x->processor.m_model_input_size);
    post("model output size: %i", x->processor.m_model_output_size);
  } 
  else {
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

void rtneural_tilde_load_model(t_rtneural_tilde *x, t_symbol s, long argc, t_atom *argv){
  (void)x;

  t_symbol* path_in = atom_getsym(argv);

  defer(x, (method)rtneural_tilde_doload_model, path_in, 0, NULL);
}  

void rtneural_tilde_reset(t_rtneural_tilde *x, long f){
  x->processor.m_model->reset();
  if(int(f)==1){
    post("model reset");
  }
}

void rtneural_tilde_bypass(t_rtneural_tilde *x, long f){
  x->bypass = f;

  post(f ? "Bypass ON" : "Bypass OFF");
}  

void rtneural_tilde_perform64(t_rtneural_tilde *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	long ch;
	double *in, *out;

  t_int small_num = x->n_in_chans;
  if(numins<x->n_in_chans){
    small_num = numins;
  }
  for (ch = 0; ch < small_num; ch++) {		// for each input channel
    if (ch<(x->n_out_chans))
      sysmem_copyptr(ins[ch], outs[ch], sizeof(double) * sampleframes);
  }

  //if not processing, just copy the input to the output
  if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
    t_int small_num = x->n_in_chans;
    if(numins<x->n_in_chans){
      small_num = numins;
    }
    for (ch = 0; ch < small_num; ch++) {		// for each input channel
      if (ch<(x->n_out_chans))
        sysmem_copyptr(ins[ch], outs[ch], sizeof(double) * sampleframes);
    }
  } else {
    if(x->trig_mode==0){
      t_int n_samps_out = x->processor.process(ins, x->input_to_nn.data(), x->in_rs.data(), x->interleaved_array.data(), x->out_temp.data(), x->outbuf.data(), x->blocksize);


      //deinterleave the output and put it in the output buffers
      for (t_int j = 0; j < x->n_out_chans; j++) {
        for(t_int i = 0; i < x->blocksize; i++) {
          outs[j][i] = (double)x->outbuf[i*x->n_out_chans+j];
        }
      }

    } else {

      x->input_model_ratio = (numins-2)/x->processor.m_model_input_size;
      if(x->input_model_ratio<1){
        x->input_model_ratio = 1;
      }

      for (t_int i = 0; i < sampleframes; ++i){
        if(ins[numins-1][i]>0.){
          x->processor.reset();
        }
        if(ins[numins-2][i]>0.){
      
          for (int l = 0; l < x->input_model_ratio; l++) {
            for (t_int j = 0; j < x->n_in_chans; ++j) {
              x->input_to_nn[j] = (float)ins[j + (l*x->n_in_chans)][i];
            }
            x->processor.process1(x->input_to_nn.data(), x->output_from_nn.data());
          }
          for (t_int j = 0; j < x->n_out_chans; ++j) {
            outs[j][i] = (double)x->output_from_nn[j];
          }
        } else {
          for (t_int j = 0; j < x->n_out_chans; ++j) {
            outs[j][i] = (double)x->output_from_nn[j];
          }
        }
      }
     }
  }
}

void rtneural_tilde_assist(t_rtneural_tilde *x, void *b, long io, long index, char *s)
{
  switch (io) {
    case 1:
      switch (index) {
        case 0:
          strncpy_zero(s, "audio/data inlet", 512);
          break;
        case 1:
          strncpy_zero(s, "trigger inlet - only works in trigger mode", 512);
          break;
        case 2:
          strncpy_zero(s, "reset inlet - only works in trigger mode", 512);
          break;
      }
      break;
    case 2:
      strncpy_zero(s, "network inference", 512);
      break;
  }
}

void rtneural_tilde_dsp64(t_rtneural_tilde *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
  if(samplerate!=x->sample_rate||maxvectorsize!=x->blocksize){
    reset_vars_and_mem(x, samplerate, maxvectorsize);
    x->processor.reset_ratio(x->ratio);
  }
	dsp_add64(dsp64, (t_object *)x, (t_perfroutine64)rtneural_tilde_perform64, 0, NULL);
}
/************************************************************************************/
