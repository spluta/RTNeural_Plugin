#include "m_pd.h"
#include "../RTN_Processor.cpp"
#include <algorithm>
#include <cmath>

static t_class *softmax_class;

class Softmax_pd {
public:
  Softmax_pd(t_int n = -1);

  t_object obj;

  t_outlet *out_top;
  t_outlet *out_rand;
  t_outlet *out_indices;
  t_outlet *out_weights;

  t_float n;
  std::vector<t_atom> indices;
  std::vector<t_atom> weights;
};

Softmax_pd::Softmax_pd(t_int n) {
  this->n = n;
  this->out_top = outlet_new(&obj, &s_float);
  this->out_rand = outlet_new(&obj, &s_float);
  this->out_indices = outlet_new(&obj, &s_list);
  this->out_weights = outlet_new(&obj, &s_list);
}

void *softmax_new(t_floatarg n) {
  Softmax_pd *x  = (Softmax_pd *)pd_new(softmax_class);
  floatinlet_new(&x->obj, &x->n);
  new (x) Softmax_pd((t_int)n);
  return (void *)x;
}

void softmax_free(Softmax_pd *x) {
  x->~Softmax_pd();
}

static void softmax_bang(Softmax_pd *x) {
  if (x->weights.empty() || (t_int)x->n == 0) {
    outlet_bang(x->out_weights);
    outlet_bang(x->out_indices);
    outlet_bang(x->out_rand);
    outlet_bang(x->out_top);
  } else {
    // if negative range or range exceeds list, output whole list
    int n = (x->n < 0 || x->n > x->weights.size()) ? x->weights.size() : x->n;
    outlet_list(x->out_weights, gensym("list"), n, x->weights.data());
    outlet_list(x->out_indices, gensym("list"), n, x->indices.data());

    t_float max = (t_float)rand() / RAND_MAX;

    unsigned long i = 0;
    for (;;) {
      max -= atom_getfloat(&x->weights[atom_getint(&x->indices[i])]);
      if (max <= 0.) {
        break;
      }
      i++;
    }
    outlet_float(x->out_rand, atom_getint(&x->indices[i]));
    outlet_float(x->out_top, atom_getint(&x->indices.front()));
  }
}

void softmax_list(Softmax_pd *x, t_symbol *s, int argc, t_atom *argv) {
  x->indices.resize(argc);
  x->weights.resize(argc);
  for (int i = 0; i < argc; i++) {
    SETFLOAT(&x->indices[i], i);
    x->weights[i] = *(argv + i);
  }
  // sort indices greatest to least based on weights (don't sort weights)
  std::sort(x->indices.begin(), x->indices.end(), [&](t_atom a, t_atom b) {
    float_t fa = atom_getfloat(&x->weights[atom_getint(&a)]);
    float_t fb = atom_getfloat(&x->weights[atom_getint(&b)]);
    return fa > fb;
  });

  softmax_bang(x);
}

extern "C" void softmax_setup(void) {
  softmax_class = class_new(
    gensym("softmax"),
    (t_newmethod)softmax_new,
    (t_method)softmax_free,
    sizeof(Softmax_pd),
    CLASS_DEFAULT,
    A_DEFFLOAT,
    0
  );
  class_addbang(softmax_class, softmax_bang);
  class_addlist(softmax_class, softmax_list);
}
