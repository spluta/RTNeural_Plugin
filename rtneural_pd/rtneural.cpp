#include "m_pd.h"
//#include <memory>
#include "../RTN_Processor.cpp"
// #include <experimental/filesystem>
#include <vector>

#include "m_pd.h"

static t_class *rtneural_class;

typedef struct _rtneural {
  t_object obj;
  t_float f;

  t_canvas *canvas; // necessary for relative paths

  // rtneural data
  t_int epochs;
  std::vector<std::vector<t_float>> in_vals;
  std::vector<std::vector<t_float>> out_vals;
  std::vector<t_int> layers_ints;
  std::vector<std::string> layers_strings;

  std::string python_path;
  
  t_float learn_rate;

  t_int bypass;

  t_int n_in_chans;
  t_int n_out_chans;

  t_atom *out_list;

  t_float ratio;
  t_float model_loaded;

  float* input_to_nn;
  float* output_from_nn;

  RTN_Processor processor;

} t_rtneural;  

void rtneural_bang(t_rtneural *obj)
{
    t_symbol *s = gensym("list");
    outlet_list(obj->obj.ob_outlet, s, obj->n_out_chans, obj->out_list);
}

void rtneural_list(t_rtneural* x, t_symbol *s, int argc, t_atom *argv) {
    if ((x->processor.m_model_loaded==0)||((t_int)x->bypass==1)) {
        return;
    }

    t_int insize = argc;
    t_int loops = insize / x->n_in_chans;

    if (insize % x->n_in_chans != 0) {
      post("input size is not a multiple of the number of input channels");
      return;
    }

    for (int i = 0; i < loops; i++) {
        for (int j = 0; j < x->n_in_chans; j++) {
            x->input_to_nn[j] = atom_getfloat(argv + i * x->n_in_chans + j);
        }
        x->processor.process1(x->input_to_nn, x->output_from_nn);
    }
      // for(int i=0; i<x->n_in_chans; i++){
      //     x->input_to_nn[i] = atom_getfloat(argv+i);
      // }
      // x->processor.process1(x->input_to_nn, x->output_from_nn);

    for(int i=0; i<x->n_out_chans; i++){
        SETFLOAT(x->out_list+i, x->output_from_nn[i]);
    }
    outlet_list(x->obj.ob_outlet, s, x->n_out_chans, x->out_list);
}

void* rtneural_new(t_floatarg n_in_chans, t_floatarg n_out_chans) {  

t_rtneural *x = (t_rtneural *)pd_new(rtneural_class);

x->canvas = canvas_getcurrent();

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, "../../RTNeural_python", absolute_path, MAXPDSTRING);
  x->python_path = absolute_path;
  post("python path: %s", x->python_path.c_str());

  x->epochs = 2000;
  x->learn_rate = 0.001;

  x->in_vals.clear();
  x->out_vals.clear();
  x->layers_ints.clear();
  x->layers_strings.clear();

  x->layers_ints.push_back(5);
  x->layers_strings.push_back("relu");
  x->layers_ints.push_back(10);
  x->layers_strings.push_back("sigmoid");

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

  x->bypass = 0;
  x->model_loaded = 0.f;

  x->input_to_nn = (float*)calloc(x->n_in_chans, sizeof(float));
  x->output_from_nn = (float*)calloc(x->n_out_chans, sizeof(float));

  x->processor.initialize(x->n_in_chans, x->n_out_chans, x->ratio);

  return (void *)x;
}

void rtneural_set_epochs(t_rtneural *x, t_floatarg n_epochs) {
  x->epochs = n_epochs;
}

void rtneural_set_layers_data(t_rtneural *x, t_symbol *s, int argc, t_atom *argv) {
  x->layers_ints.clear();
  x->layers_strings.clear();
  for (int i = 0; i < argc / 2; i++) {
    //x->layers_data[atom_getfloat(argv + i * 2)] = atom_getsymbol(argv + i * 2 + 1)->s_name;
    x->layers_ints.push_back(atom_getfloat(argv + i * 2));
    x->layers_strings.push_back(atom_getsymbol(argv + i * 2 + 1)->s_name);
  }
}

void rtneural_set_learn_rate(t_rtneural *x, t_floatarg learn_rate) {
  x->learn_rate = learn_rate;
}

void rtneural_clear_points(t_rtneural *x, t_symbol *s, int argc, t_atom *argv) {
  x->in_vals.clear();
  x->out_vals.clear();
  post("cleared in and out vals");
}

void rtneural_post_points(t_rtneural *x, t_symbol *s, int argc, t_atom *argv) {
  post("inputs:");
  std::string temp_str;
  for (size_t i = 0; i < x->in_vals.size(); i++) {
    temp_str.clear();
    temp_str += "point ";
    temp_str += std::to_string(i);
    temp_str += " ";
    for (size_t j = 0; j < x->in_vals[i].size(); j++) {
      temp_str += std::to_string(x->in_vals[i][j]);
      if (j < x->in_vals[i].size() - 1) {
        temp_str += ", ";
      }
    }
    post(temp_str.c_str());
  }
  post("outputs:");
  for (size_t i = 0; i < x->out_vals.size(); i++) {
    temp_str.clear();
    temp_str += "point ";
    temp_str += std::to_string(i);
    temp_str += " ";
    for (size_t j = 0; j < x->out_vals[i].size(); j++) {
      temp_str += std::to_string(x->out_vals[i][j]);
      if (j < x->out_vals[i].size() - 1) {
        temp_str += ", ";
      }
    }
    post(temp_str.c_str());
  }
}

void rtneural_remove_point(t_rtneural *x, t_floatarg index_in) {
  t_int index = t_int(index_in);
  if (index < 0 || index >= x->in_vals.size()) {
    post("index out of range");
    return;
  }
  x->in_vals.erase(x->in_vals.begin() + index);
  x->out_vals.erase(x->out_vals.begin() + index);
}

void rtneural_add_input(t_rtneural *x, t_symbol *s, int argc, t_atom *argv) {
  std::vector<t_float> in_temp;
  for (int i = 0; i < argc; i++) {
    in_temp.push_back(atom_getfloat(argv + i));
  }
  x->in_vals.push_back(in_temp);
}

void rtneural_add_output(t_rtneural *x, t_symbol *s, int argc, t_atom *argv) {
  std::vector<t_float> out_temp;
  for (int i = 0; i < argc; i++) {
    out_temp.push_back(atom_getfloat(argv + i));
  }
  x->out_vals.push_back(out_temp);
}

void rtneural_write_json(t_rtneural *x, t_symbol *s){
  nlohmann::json data;

  data["epochs"] = x->epochs;
  data["learn_rate"] = x->learn_rate;

  data["layers_data"] = nlohmann::json::array_t();
  for (size_t i = 0; i < x->layers_ints.size(); i++) {
    data["layers_data"].push_back({x->layers_ints[i], x->layers_strings[i]});
  }

  data["in_vals"] = nlohmann::json::array_t();
  for(size_t i=0; i<x->in_vals.size(); i++){
    data["in_vals"].push_back(nlohmann::json::array_t());
    for(size_t j=0; j<x->in_vals[i].size(); j++){
      data["in_vals"][i].push_back(x->in_vals[i][j]);
    }
  }

  data["out_vals"] = nlohmann::json::array_t();
  for(size_t i=0; i<x->out_vals.size(); i++){
    data["out_vals"].push_back(nlohmann::json::array_t());
    for(size_t j=0; j<x->out_vals[i].size(); j++){
      data["out_vals"][i].push_back(x->out_vals[i][j]);
    }
  }

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  post(absolute_path);

  std::string parent_directory = absolute_path;
  size_t pos = parent_directory.find_last_of("/\\");
  if (pos != std::string::npos) {
    parent_directory = parent_directory.substr(0, pos);
  }
  if (!std::filesystem::is_directory(parent_directory.c_str())) {
    post("The directory does not exist or is not a directory");
    return;
  }

  std::ofstream output_file(absolute_path);
  if (!output_file.is_open())  {
    post("Failed to open output file");
  } else {
    output_file << data;
    output_file.close();
    post("writing output file");
  }
}

void rtneural_train_model(t_rtneural *x, t_symbol *s) {
  post("not yet implemented");
  // char absolute_path[MAXPDSTRING] = { 0 };
  // canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  // std::string cmd = "cd "
  //   + x->python_path
  // #ifdef _WIN32
  //   + "; venv\\bin\\activate.bat; "
  //   + "python MLP_control\\mlp_control_train_convert.py -f "
  // #else
  //   + "; . venv/bin/activate; "
  //   + "python MLP_control/mlp_control_train_convert.py -f "
  // #endif
  //   + absolute_path;
  // post(cmd.c_str());

  // std::unique_ptr<FILE, void(*)(FILE *)> pipe(popen(cmd.c_str(), "r"),
  //   [](FILE *f) -> void {
  //     std::ignore = pclose(f);
  //   }
  // );
  // if (!pipe) {
  //   post("Failed to open pipe");
  //   return;
  // }
  // char buffer[128];
  // try {
  //   while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
  //     post(buffer);
  //   }
  // } catch (...) {
  //   post("Failed to read output");
  //   return;
  // }
}

void rtneural_load_model(t_rtneural *x, t_symbol *s){
  (void)x;

  post("loading model: ");
  post(s->s_name);

  char absolute_path[MAXPDSTRING] = { 0 };
  canvas_makefilename(x->canvas, s->s_name, absolute_path, MAXPDSTRING);

  t_int test = x->processor.load_model(absolute_path, 1);
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

void rtneural_free (t_rtneural* obj) {

}

void rtneural_reset(t_rtneural *x, t_symbol *s, int argc, t_atom *argv){
  t_int temp = atom_getint(argv);
  x->processor.reset();
  if((int)temp==1){
    post("model reset");
  }
}

void rtneural_bypass(t_rtneural *x, t_float f){
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
        0, sizeof(t_rtneural),
        CLASS_DEFAULT,
        A_DEFFLOAT, A_DEFFLOAT, 0);

    class_addbang(rtneural_class, rtneural_bang);
    class_addlist(rtneural_class, rtneural_list);
    class_addmethod(rtneural_class, (t_method)rtneural_load_model, gensym("load_model"), A_SYMBOL, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_write_json, gensym("write_json"), A_SYMBOL, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_train_model, gensym("train_model"), A_SYMBOL, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_bypass, gensym("bypass"), A_FLOAT, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_reset, gensym("reset"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_set_epochs, gensym("set_epochs"), A_DEFFLOAT, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_set_layers_data, gensym("set_layers_data"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_set_learn_rate, gensym("set_learn_rate"), A_DEFFLOAT, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_clear_points, gensym("clear_points"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_post_points, gensym("post_points"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_add_input, gensym("add_input"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_add_output, gensym("add_output"), A_GIMME, 0);
    class_addmethod(rtneural_class, (t_method)rtneural_remove_point, gensym("remove_point"), A_DEFFLOAT, 0);
}

