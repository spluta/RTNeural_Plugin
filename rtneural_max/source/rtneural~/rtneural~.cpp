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
typedef struct _rtneural {
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

  float* input_to_nn;
  float* output_from_nn;

	RTN_Processor processor;

  float* interleaved_array;
  float* outbuf;

  float *in_rs;
  float *out_temp;
} t_rtneural; 


// prototypes
void	*rtneural_new(t_symbol *s, long argc, t_atom *argv);
void	rtneural_free(t_rtneural *x);
void 	rtneural_load_model(t_rtneural *x, t_symbol s, long argc, t_atom *argv);
void 	rtneural_bypass(t_rtneural *x, long f);

void rtneural_perform64(t_rtneural *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void rtneural_dsp64(t_rtneural *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void reset_vars_and_mem(t_rtneural *x, float sample_rate, t_int blocksize);
long rtneural_multichanneloutputs(t_rtneural *x, long outletindex);

// globals
static t_class *rtneural_class = NULL;

/************************************************************************************/

void ext_main(void *r)
{
  t_class	*c = class_new("rtneural~",
      (method)rtneural_new,
      (method)rtneural_free, sizeof(t_rtneural),
      (method)NULL,
      A_GIMME, 0);

  class_addmethod(c, (method)rtneural_load_model, "load_model", A_GIMME, 0);
  class_addmethod(c, (method)rtneural_bypass, "bypass", A_LONG, 0);
  class_addmethod(c, (method)rtneural_dsp64, "dsp64",	A_CANT, 0);  

  class_addmethod(c, (method)rtneural_multichanneloutputs, "multichanneloutputs", A_CANT, 0);

  CLASS_ATTR_LONG(c, "trig_mode", 0, t_rtneural, trig_mode);
  CLASS_ATTR_FILTER_CLIP(c, "trig_mode", 0, 1);

  class_dspinit(c);
	class_register(CLASS_BOX, c);
	rtneural_class = c;
}


/************************************************************************************/
// object Creation Method
void *rtneural_new(t_symbol *s, long argc, t_atom *argv)
{
	t_rtneural *x = (t_rtneural *)object_alloc(rtneural_class);
	
  post("new rtneural~ object");

  float n_in_chans = atom_getfloatarg(0, argc, argv);
  float n_out_chans = atom_getfloatarg(1, argc, argv);
  float nn_sample_rate = atom_getfloatarg(2, argc, argv);
  float trig_mode = atom_getfloatarg(3, argc, argv);

  if(n_in_chans<1.f){
    n_in_chans = 1.f;
  }
  if(n_out_chans<1.f){
    n_out_chans = 1.f;
  }
  x->n_in_chans = (t_int)n_in_chans;
  x->n_out_chans = (t_int)n_out_chans;

  x->input_to_nn = (float*)sysmem_newptr(x->n_in_chans*sizeof(float));
	x->output_from_nn = (float*)sysmem_newptr(x->n_out_chans*sizeof(float));

  for(int i=0; i<x->n_in_chans; i++){
    x->input_to_nn[i] = 0.f;
  }
  for(int i=0; i<x->n_out_chans; i++){
    x->output_from_nn[i] = 0.f;
  }
  
  if(nn_sample_rate<=0.f){
    nn_sample_rate = sys_getsr();
  }
  x->nn_sample_rate = nn_sample_rate;

  if (trig_mode!=1.f) {
    trig_mode = 0.f;
  }
  x->trig_mode = (long)trig_mode;

  x->bypass = 0;
  x->model_loaded = 0.f;

  reset_vars_and_mem(x, sys_getsr(), sys_getblksize());

  x->processor.initialize(x->n_in_chans, x->n_out_chans, x->ratio);
	
	dsp_setup((t_pxobject *)x, 1);
	x->m_obj.z_misc |= Z_NO_INPLACE | Z_MC_INLETS;
	outlet_new((t_object *)x, "multichannelsignal");
	return x;
}

long rtneural_multichanneloutputs(t_rtneural *x, long outletindex)
{
  return x->n_out_chans; 
}

void reset_vars_and_mem(t_rtneural *x, float sample_rate, t_int blocksize){
  post("resetting nn sample rate and block size");

  x->sample_rate = sample_rate;
  x->blocksize = blocksize;
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  x->ratio = 1.f;
  x->ratio = x->nn_sample_rate/x->sample_rate;
  post("ratio: %f", x->ratio);

  t_int rs_size = t_int(ceil(x->nn_sample_rate/x->control_rate));

  t_int in_size = x->blocksize*x->n_in_chans;
  t_int in_rs_size = rs_size*x->n_in_chans;
  t_int out_temp_size = rs_size*x->n_out_chans; 
  t_int out_buf_size = x->blocksize*x->n_out_chans;

  x->sample_rate = sample_rate;
  x->blocksize = blocksize;
  x->control_rate = x->sample_rate/t_float(x->blocksize);

  x->ratio = 1.f;

  sysmem_freeptr(x->interleaved_array);
  sysmem_freeptr(x->in_rs);
  sysmem_freeptr(x->out_temp);
  sysmem_freeptr(x->outbuf);

  x->interleaved_array = (float *)sysmem_newptr(in_size * sizeof(float));
  x->in_rs = (float *)sysmem_newptr(in_rs_size * sizeof(float));
  x->out_temp = (float *)sysmem_newptr(out_temp_size * sizeof(float));
  x->outbuf = (float *)sysmem_newptr(out_buf_size * sizeof(float));

}

void rtneural_free (t_rtneural* x) {
  z_dsp_free((t_pxobject *)x);
  x->processor.~RTN_Processor();

  sysmem_freeptr(x->interleaved_array);
  sysmem_freeptr(x->in_rs);
  sysmem_freeptr(x->out_temp);
  sysmem_freeptr(x->outbuf);
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

void rtneural_doload_model(t_rtneural *x, t_symbol *path_in){
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

void rtneural_load_model(t_rtneural *x, t_symbol s, long argc, t_atom *argv){
  (void)x;

  t_symbol* path_in = atom_getsym(argv);

  defer(x, (method)rtneural_doload_model, path_in, 0, NULL);
}  

void rtneural_bypass(t_rtneural *x, long f){
  x->bypass = f;

  post(f ? "Bypass ON" : "Bypass OFF");
}  

void rtneural_perform64(t_rtneural *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	long ch;
	double *in, *out;

  //if not processing, just copy the input to the output
  if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
    //t_int small_num = min(x->n_in_chans, numins);
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
      t_int n_samps_out = x->processor.process(ins, x->in_rs, x->interleaved_array, x->out_temp, x->outbuf, x->blocksize);

      //post("interleaved: %f %f %f %f %f %f ", x->outbuf[0], x->outbuf[1], x->outbuf[2], x->outbuf[3], x->outbuf[4], x->outbuf[5]);

      //deinterleave the output and put it in the output buffers
      for (t_int j = 0; j < x->n_out_chans; j++) {
        for(t_int i = 0; i < x->blocksize; i++) {
          outs[j][i] = (double)x->outbuf[i*x->n_out_chans+j];
        }
      }

    } else {
      for (t_int i = 0; i < sampleframes; ++i){
        if(ins[numins-1][i]>0.){
          for (t_int j = 0; j < x->n_in_chans; ++j) {
            x->input_to_nn[j] = (float)ins[j][i];
          }
          x->processor.process1(x->input_to_nn, x->output_from_nn);
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

void rtneural_dsp64(t_rtneural *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
  if(samplerate!=x->sample_rate||maxvectorsize!=x->blocksize){
    reset_vars_and_mem(x, samplerate, maxvectorsize);
    x->processor.reset_ratio(x->ratio);
  }
	dsp_add64(dsp64, (t_object *)x, (t_perfroutine64)rtneural_perform64, 0, NULL);
}
/************************************************************************************/
