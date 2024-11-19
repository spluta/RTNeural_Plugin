#include "m_pd.h"
//#include <memory>
#include "../RTN_Processor.cpp"
#include <vector>

#include "m_pd.h"

static t_class *rtneural_class;

typedef struct _rtneural {
  t_object obj;
  t_float f;
  t_float freq;
  t_float sample_rate;
  t_int blocksize;
  t_float control_rate;
  t_float nn_sample_rate;
  t_int bypass;
  t_int n_in_chans;
  t_int n_out_chans;

    t_atom *out_list;

  t_float ratio;
  t_float model_loaded;

//   std::vector<const t_float*> in_vec;
//   std::vector< std::vector<t_float> > outVecs;
    float* input_to_nn;
    float* output_from_nn;

    RTN_Processor processor;

  float* interleaved_array;
  float* outbuf;

  float *in_rs;
  float *out_temp;\
} rtneural_t;  

void rtneural_bang(rtneural_t *obj)
{
    t_symbol *s = gensym("list");
    outlet_list(obj->obj.ob_outlet, s, obj->n_out_chans, obj->out_list);
}

void rtneural_list(rtneural_t* obj, t_symbol *s, int argc, t_atom *argv) {
    if ((obj->processor.m_model_loaded==0)||((t_int)obj->bypass==1)) {
        return;
    }
    for(int i=0; i<obj->n_in_chans; i++){
        obj->input_to_nn[i] = atom_getfloat(argv+i);
    }
    obj->processor.process1(obj->input_to_nn, obj->output_from_nn);
    for(int i=0; i<obj->n_out_chans; i++){
        SETFLOAT(obj->out_list+i, obj->output_from_nn[i]);
    }
    outlet_list(obj->obj.ob_outlet, s, obj->n_out_chans, obj->out_list);
}

void* rtneural_new(t_floatarg n_in_chans, t_floatarg n_out_chans) {  

rtneural_t *x = (rtneural_t *)pd_new(rtneural_class);

  post("%f %f", n_in_chans, n_out_chans);

    outlet_new(&x->obj, &s_list);

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

  x->input_to_nn = (float*)calloc(x->n_in_chans, sizeof(float));
    x->output_from_nn = (float*)calloc(x->n_out_chans, sizeof(float));

  //this is needed to handle resampling of audio when the sample rate is not the same as that at which the model was trained
  t_int in_size = x->n_in_chans;
  t_int out_buf_size = x->n_out_chans; 

  x->processor.initialize(x->n_in_chans, x->n_out_chans, x->ratio);

  x->interleaved_array = (float*)calloc(in_size, sizeof(float));
  x->outbuf = (float*)calloc(out_buf_size, sizeof(float));

  return (void *)x;
}

void rtneural_load_model(rtneural_t *x, t_symbol s){
  (void)x;

  post("loading model: ");
  post(s.s_name);

  t_int test = x->processor.load_model(s.s_name, 1);
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

void rtneural_free (rtneural_t* obj) {
  free(obj->interleaved_array);
  free(obj->outbuf);
	// outlet_free(obj->signal_out);
}

void rtneural_bypass(rtneural_t *x, t_float f){
  x->bypass = t_int(f);

  post(f ? "Bypass ON" : "Bypass OFF");
}  

#if defined(_LANGUAGE_C_PLUS_PLUS) || defined(__cplusplus)
extern "C" {
  void rtneural_setup(void);
}
#endif

void rtneural_setup(void) {
  rtneural_class = class_new(gensym("rtneural"),
        (t_newmethod)rtneural_new,
        0, sizeof(rtneural_t),
        CLASS_DEFAULT,
        A_DEFFLOAT, A_DEFFLOAT, 0);

  class_addbang(rtneural_class, rtneural_bang);
    class_addlist(rtneural_class, rtneural_list);
    class_addmethod(rtneural_class, (t_method)rtneural_load_model, gensym("load_model"), A_SYMBOL, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_bypass, gensym("bypass"), A_FLOAT, 0);
}

// static t_class *rtneural_class;






// // t_int* rtneural_perform (t_int* args) {
// //   rtneural_t* obj = (rtneural_t*)args[1];
// //   t_sample    *in =      (t_sample *)(args[2]);
// //   t_sample* out = (t_sample *)args[3];
// //   t_int n_samps = (t_int)args[4];


// //   if ((obj->processor.m_model_loaded==0)||((t_int)obj->bypass==1)) {
// //     for (t_int i = 0; i < obj->blocksize; ++i) {
// //       out[i] = in[i];
// //     }
// //   } else {
// //     obj->in_vec[0] = in;
// //     for (t_int j = 1; j < obj->n_in_chans; j++) {
// //       obj->in_vec[j] = in + j * static_cast<t_int>(obj->blocksize);
// //     }

// //     t_int n_samps_out = obj->processor.process(obj->in_vec, obj->in_rs, obj->interleaved_array, obj->out_temp, obj->outbuf, obj->blocksize);

// //     //deinterleave the output and put it in the output buffers
// //     for(t_int i = 0; i < n_samps_out; i++) {
// //       for (t_int j = 0; j < obj->n_out_chans; j++) {
// //         out[j*obj->blocksize+ i] = obj->outbuf[i*obj->n_out_chans+j];
// //       }
// //     }
// //   }

// //   return (t_int *) (args + 5);
// // }

// // void rtneural_float(rtneural_t* obj, t_float f) {
// //   post("list: %f", f);
// // }



// #if defined(_LANGUAGE_C_PLUS_PLUS) || defined(__cplusplus)
// extern "C" {
//   void rtneural_setup(void);
// }
// #endif

// void rtneural_setup(void) {
//   rtneural_class = class_new(gensym("rtneural"),
//     (t_newmethod)rtneural_new, 
//     (t_method)rtneural_free,
//     sizeof(rtneural_t), 
//     CLASS_DEFAULT, A_DEFFLOAT, A_DEFFLOAT, 0);
  
//   class_addbang(rtneural_class, rtneural_bang);
//   // class_addfloat(rtneural_class, rtneural_float);
// //   class_addmethod(rtneural_class, (t_method)rtneural_load_model, gensym("load_model"), A_SYMBOL, 0);

// //   class_addmethod(rtneural_class, (t_method)rtneural_bypass, gensym("bypass"), A_FLOAT, 0);
// // //   class_addfloat(rtneural_class, rtneural_float);
// //   class_addlist(rtneural_class, rtneural_list);
// }



