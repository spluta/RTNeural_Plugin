/**
	@file
	collect - collect numbers and operate on them.
			- demonstrates use of C++ and the STL in a Max external
			- also demonstrates use of a mutex for thread safety
			- on Windows, demonstrate project setup for static linking to the Microsoft Runtime

	@ingroup	examples

	Copyright 2009 - Cycling '74
	Timothy Place, tim@cycling74.com
*/

#include "ext.h"
#include "ext_obex.h"
#include "ext_strings.h"
#include "ext_common.h"
#include "ext_systhread.h"

#include "../RTN_Processor.cpp"


#include <vector>
using namespace std;

// a wrapper for cpost() only called for debug builds on Windows
// to see these console posts, run the DbgView program (part of the SysInternals package distributed by Microsoft)
#if defined( NDEBUG ) || defined( MAC_VERSION )
#define DPOST
#else
#define DPOST cpost
#endif

// max object instance data
typedef struct _rtneural {
  t_object m_obj;
  void	*outlet;
  t_systhread_mutex	mutex;

  float f;
  float freq;
  float sample_rate;
  t_int blocksize;
  float control_rate;
  float nn_sample_rate;
  t_int bypass;
  t_int n_in_chans;
  t_int n_out_chans;

  t_atom *out_list;

  float ratio;
  float model_loaded;

	float* input_to_nn;
	float* output_from_nn;

	RTN_Processor processor;

  float *in_rs;
  float *out_temp;\
} rtneural_t; 


// prototypes
void	*rtneural_new(t_symbol *s, long argc, t_atom *argv);
void	rtneural_free(rtneural_t *x);
void	rtneural_bang(rtneural_t *x);
void	rtneural_list(rtneural_t *x, t_symbol *s, long argc, t_atom *argv);
void 	rtneural_write_json(rtneural_t *x, t_symbol s);
void 	rtneural_load_model(rtneural_t *x, t_symbol s, long argc, t_atom *argv);
void 	rtneural_bypass(rtneural_t *x, long f);

// globals
//static t_class	*s_collect_class = NULL;
static t_class *rtneural_class = NULL;

/************************************************************************************/

void ext_main(void *r)
{
	  t_class	*c = class_new("rtneural",
        (method)rtneural_new,
        (method)rtneural_free, sizeof(rtneural_t),
        (method)NULL,
        A_GIMME, 0);

	class_addmethod(c, (method)rtneural_bang,"bang",0);
	class_addmethod(c, (method)rtneural_list,	"list",	A_GIMME,0);
  class_addmethod(c, (method)rtneural_load_model, "load_model", A_SYM, 0);
  class_addmethod(c, (method)rtneural_write_json, "write_json", A_SYM, 0);
  class_addmethod(c, (method)rtneural_bypass, "bypass", A_LONG, 0);
	class_register(CLASS_BOX, c);
	rtneural_class = c;
}


/************************************************************************************/
// object Creation Method

void* rtneural_new(t_symbol *s, long argc, t_atom *argv) {  

	rtneural_t *x;
	x = (rtneural_t *)object_alloc(rtneural_class);

	if (x) {
		//systhread_mutex_new(&x->c_mutex, 0);
		x->outlet = outlet_new(x, NULL);

		float n_in_chans = atom_getfloat(argv);
		float n_out_chans = atom_getfloat(argv+1);

		if(n_in_chans<1.f){
			n_in_chans = 1.f;
		}
		if(n_out_chans<1.f){
			n_out_chans = 1.f;
		}
		x->n_in_chans = t_int(n_in_chans);
		x->n_out_chans = t_int(n_out_chans);
		x->out_list = (t_atom *)getbytes(n_out_chans * sizeof(t_atom));

		x->blocksize = 1;

		x->bypass = 0;
		x->model_loaded = 0.f;

		x->ratio = 1.f;

		x->input_to_nn = (float*)sysmem_newptr(x->n_in_chans*sizeof(float));
		x->output_from_nn = (float*)sysmem_newptr(x->n_out_chans*sizeof(float));

		//this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
		t_int in_size = x->n_in_chans;
		t_int out_buf_size = x->n_out_chans; 

		x->processor.initialize(x->n_in_chans, x->n_out_chans, x->ratio);
	}
  return x;
}

void rtneural_free (rtneural_t* x) {
  sysmem_freeptr(x->input_to_nn);
  sysmem_freeptr(x->output_from_nn);
}

void rtneural_bang(rtneural_t *x)
{
    t_symbol *s = gensym("list");
    outlet_list(x->outlet, s, x->n_out_chans, x->out_list);
}

void rtneural_list(rtneural_t *x, t_symbol *s, long argc, t_atom *argv) {
    if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
        return;
    }
    for(int i=0; i<x->n_in_chans; i++){
        x->input_to_nn[i] = atom_getfloat(argv+i);
    }
    x->processor.process1(x->input_to_nn, x->output_from_nn);
    for(int i=0; i<x->n_out_chans; i++){
        atom_setfloat(x->out_list+i, x->output_from_nn[i]);
    }
    outlet_list(x->outlet, NULL, x->n_out_chans, x->out_list);
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
//void myobject_message(t_myobject *x, t_symbol *s, long argc, t_atom *argv);
void rtneural_load_model(rtneural_t *x, t_symbol s, long argc, t_atom *argv){
  (void)x;

	t_symbol* path = atom_getsym(argv);

	post("message selector is %s",s.s_name);
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

/************************************************************************************/
