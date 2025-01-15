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

#include "../RTN_Processor.cpp"

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
} rtneural_t; 


// prototypes
void	*rtneural_new(t_symbol *s, long argc, t_atom *argv);
void	rtneural_free(rtneural_t *x);
void 	rtneural_write_json(rtneural_t *x, t_symbol s);
void 	rtneural_load_model(rtneural_t *x, t_symbol s, long argc, t_atom *argv);
void 	rtneural_bypass(rtneural_t *x, long f);

void rtneural_perform64(rtneural_t *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void rtneural_dsp64(rtneural_t *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void reset_vars_and_mem(rtneural_t *x, float sample_rate, t_int blocksize);

// globals
static t_class *rtneural_class = NULL;

/************************************************************************************/

void ext_main(void *r)
{
  t_class	*c = class_new("rtneural~",
      (method)rtneural_new,
      (method)rtneural_free, sizeof(rtneural_t),
      (method)NULL,
      A_GIMME, 0);

  class_addmethod(c, (method)rtneural_load_model, "load_model", A_SYM, 0);
  class_addmethod(c, (method)rtneural_write_json, "write_json", A_SYM, 0);
  class_addmethod(c, (method)rtneural_bypass, "bypass", A_LONG, 0);
  class_addmethod(c, (method)rtneural_dsp64, "dsp64",	A_CANT, 0);  

  CLASS_ATTR_LONG(c, "trig_mode", 0, rtneural_t, trig_mode);
  CLASS_ATTR_FILTER_CLIP(c, "trig_mode", 0, 1);

  class_dspinit(c);
	class_register(CLASS_BOX, c);
	rtneural_class = c;
}


/************************************************************************************/
// object Creation Method
void *rtneural_new(t_symbol *s, long argc, t_atom *argv)
{
	rtneural_t *x = (rtneural_t *)object_alloc(rtneural_class);
	
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
    //x->inVecSmall.push_back(0.f);
  }
  for(int i=0; i<x->n_out_chans; i++){
    x->output_from_nn[i] = 0.f;
    //x->outVecSmall.push_back(0.f);
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

void reset_vars_and_mem(rtneural_t *x, float sample_rate, t_int blocksize){
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

void rtneural_free (rtneural_t* x) {
  z_dsp_free((t_pxobject *)x);
  x->processor.~RTN_Processor();

  sysmem_freeptr(x->interleaved_array);
  sysmem_freeptr(x->in_rs);
  sysmem_freeptr(x->out_temp);
  sysmem_freeptr(x->outbuf);
}

void rtneural_write_json(rtneural_t *x, t_symbol s){
  	nlohmann::json data;

    data["Some String"] = "Hello World";
    data["Some Number"] = 12345;
    data["Empty Array"] = nlohmann::json::array_t();
    data["Array With Items"] = { true, "Hello", 3.1415 };
    data["Empty Array"] = nlohmann::json::array_t();
    data["xect With Items"] = nlohmann::json::object_t({{"Key", "Value"}, {"Day", true}});
    data["Empty xect"] = nlohmann::json::object_t();

    std::ofstream output_file(s.s_name);
    if (!output_file.is_open())  {
        post("Failed to open output file");
    } else {
        output_file << data;
        output_file.close();
        post("writing output file");
    }
}

void rtneural_load_model(rtneural_t *x, t_symbol s, long argc, t_atom *argv){

	t_symbol* path = atom_getsym(argv);

  post("loading model: ");
  post(path->s_name);

  t_int test = x->processor.load_model(path->s_name, 1);
  if(test==1){
    x->model_loaded = 1;
  }

  post("model input size: %i", x->processor.m_model_input_size);
  post("model output size: %i", x->processor.m_model_output_size);

  if(x->processor.m_model_input_size!=x->n_in_chans){
    post("error: model input size does not match the number of input channels");
    post("disabling model");
    x->model_loaded = 0;
  }
  if(x->processor.m_model_output_size!=x->n_out_chans){
    post("error: model output size does not match the number of output channels");
    post("disabling model");
    x->model_loaded = 0;
  }
}  

void rtneural_bypass(rtneural_t *x, long f){
  x->bypass = f;

  post(f ? "Bypass ON" : "Bypass OFF");
}  

void rtneural_perform64(rtneural_t *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	long ch;
	double *in, *out;

  // if not processing, just copy the input to the output
  if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
    for (ch = 0; ch < numins; ch++) {		// for each input channel
      if (ch<x->n_out_chans)
        sysmem_copyptr(ins[ch], outs[ch], sizeof(double) * sampleframes);
    }
  } else {
    if(x->trig_mode==0){
      t_int n_samps_out = x->processor.process(ins, x->in_rs, x->interleaved_array, x->out_temp, x->outbuf, x->blocksize);

      //deinterleave the output and put it in the output buffers
      for (t_int j = 0; j < x->n_out_chans; j++) {
        for(t_int i = 0; i < n_samps_out; i++) {
          outs[j][i] = (double)x->outbuf[j*n_samps_out+i];
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

void rtneural_dsp64(rtneural_t *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
  if(samplerate!=x->sample_rate||maxvectorsize!=x->blocksize){
    reset_vars_and_mem(x, samplerate, maxvectorsize);
    x->processor.reset_ratio(x->ratio);
  }
	dsp_add64(dsp64, (t_object *)x, (t_perfroutine64)rtneural_perform64, 0, NULL);
}
/************************************************************************************/
