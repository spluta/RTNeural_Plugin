#include "m_pd.h"
//#include <memory>
#include "../RTN_Processor.cpp"
#include <map>
#include <string>
#include <vector>

#include "m_pd.h"

static t_class *rtneural_class;

typedef struct _rtneural {
  t_object obj;
  t_canvas *canvas; // necessary for relative paths

  // rtneural data
  t_int epochs;
  std::vector<t_float> in_vals;
  std::map<t_int, std::string> layers_data;
  t_float learn_rate;
  std::vector<t_float> out_vals;

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

    float* input_to_nn;
    float* output_from_nn;

    RTN_Processor processor;

  float* interleaved_array;
  float* outbuf;

  float *in_rs;
  float *out_temp;
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
  x->canvas = canvas_getcurrent();

  x->epochs = 0;
  x->learn_rate = 0;

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

void rtneural_set_epochs(rtneural_t *x, t_floatarg n_epochs) {
  x->epochs = n_epochs;
}

void rtneural_set_in_vals(rtneural_t *x, t_symbol *s, int argc, t_atom *argv) {
  x->in_vals.clear();
  for (int i = 0; i < argc; i++) {
    x->in_vals.push_back(atom_getfloat(argv + i));
  }
}

void rtneural_set_layers_data(rtneural_t *x, t_symbol *s, int argc, t_atom *argv) {
  x->layers_data.clear();
  for (int i = 0; i < argc / 2; i++) {
    x->layers_data[atom_getfloat(argv + i * 2)] = atom_getsymbol(argv + i * 2 + 1)->s_name;
  }
}

void rtneural_set_learn_rate(rtneural_t *x, t_floatarg learn_rate) {
  x->learn_rate = learn_rate;
}

void rtneural_set_out_vals(rtneural_t *x, t_symbol *s, int argc, t_atom *argv) {
  x->out_vals.clear();
  for (int i = 0; i < argc; i++) {
    x->out_vals.push_back(atom_getfloat(argv + i));
  }
}

void rtneural_write_json(rtneural_t *x, t_symbol *s){
  nlohmann::json data;

  data["epochs"] = x->epochs;
  data["in_vals"] = nlohmann::json::array_t();
  for (size_t i = 0; i < x->in_vals.size() / x->n_in_chans; i++) {
    data["in_vals"].push_back(nlohmann::json::array_t());
    for (size_t j = 0; j < x->n_in_chans; j++) {
      data["in_vals"][i].push_back(x->in_vals[i * x->n_in_chans + j]);
    }
  }
  data["layers_data"] = x->layers_data;
  post("%f", x->learn_rate);
  data["learn_rate"] = x->learn_rate;
  data["out_vals"] = nlohmann::json::array_t();
  for (size_t i = 0; i < x->out_vals.size() / x->n_out_chans; i++) {
    data["out_vals"].push_back(nlohmann::json::array_t());
    for (size_t j = 0; j < x->n_out_chans; j++) {
      data["out_vals"][i].push_back(x->out_vals[i * x->n_out_chans + j]);
    }
  }
  // data["Some String"] = "Hello World";
  // data["Some Number"] = 12345;
  // data["Empty Array"] = nlohmann::json::array_t();
  // data["Array With Items"] = { true, "Hello", 3.1415 };
  // data["Empty Array"] = nlohmann::json::array_t();
  // data["Object With Items"] = nlohmann::json::object_t({{"Key", "Value"}, {"Day", true}});
  // data["Empty Object"] = nlohmann::json::object_t();

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  std::ofstream output_file(absolute_path);
  if (!output_file.is_open())  {
    post("Failed to open output file");
  } else {
    output_file << data;
    output_file.close();
    post("writing output file");
  }
}

void rtneural_load_model(rtneural_t *x, t_symbol *s){
  (void)x;

  post("loading model: ");
  post(s->s_name);

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  t_int test = x->processor.load_model(absolute_path, 1);
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

void rtneural_bypass(rtneural_t *x, t_floatarg f){
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
  class_addmethod(rtneural_class, (t_method)rtneural_load_model, gensym("load_model"), A_DEFSYMBOL, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_write_json, gensym("write_json"), A_DEFSYMBOL, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_bypass, gensym("bypass"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_set_epochs, gensym("set_epochs"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_set_in_vals, gensym("set_in_vals"), A_GIMME, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_set_layers_data, gensym("set_layers_data"), A_GIMME, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_set_learn_rate, gensym("set_learn_rate"), A_DEFFLOAT, 0);
  class_addmethod(rtneural_class, (t_method)rtneural_set_out_vals, gensym("set_out_vals"), A_GIMME, 0);
}
