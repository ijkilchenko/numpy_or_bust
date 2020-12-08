#include <math.h>

#include <iostream>
#include <regex>
#include <string>
#include <typeinfo>
#include <vector>

using namespace std;

// Good diagram:
// https://engmrk.com/wp-content/uploads/2018/09/Image-Architecture-of-Convolutional-Neural-Network.png

class Layer {
 public:
  virtual ~Layer() = default;

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> x);
  vector<double> h(vector<double> x);

  // Helper functions
  static void rand_init(vector<vector<double>>& matrix, int height, int width) {
    srand(time(NULL));  // Remove to stop seeding rand()

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // use numbers between -100 and 100
        double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
        n = n * 200 - 100;
        matrix[i][j] = n;  // (possibly) change to use float to save memory
      }
    }
  }

  static void rand_init(vector<double> matrix, int length) {
    srand(time(NULL));  // Remove to stop seeding rand()
    for (int i = 0; i < length; i++) {
      // use numbers between -100 and 100
      double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
      n = n * 200 - 100;
      matrix[i] = n;
    }
  }

  vector<vector<double>> static add_matrices(vector<vector<double>> a, vector<vector<double>> b) {
    vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));

    for (int i = 0; i < a.size(); i++) {
      vector<double> c_column;
      for (int j = 0; j < a[0].size(); j++) {
        c[i][j] = (a[i][j] + b[i][j]);
      }
    }

    return c;
  }
};

class Conv : public Layer {
 public:
  int num_input_channels;
  int num_filters;
  vector<int> size_per_filter;
  vector<int> stride_per_filter;

  vector<vector<vector<double>>> filters;
  // TODO: Add a bias per filter.
  Conv(int num_input_channels, int num_filters, vector<int> size_per_filter, vector<int> stride_per_filter) {
    // TODO: Check if there is a better way to save these.
    this->num_input_channels = num_input_channels;
    this->num_filters = num_filters;
    this->size_per_filter = size_per_filter;
    this->stride_per_filter = stride_per_filter;

    for (int i = 0; i < num_filters; i++) {
      // Filters are square
      int height = size_per_filter[i];
      int width = size_per_filter[i];

      vector<vector<double>> filter(height, vector<double>(width, 0));
      Layer().rand_init(filter, height, width);
      this->filters.push_back(filter);
    }
  }

  // TODO: Write a test for this function if needed.
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    // Input and output is num_channels x height x width
    // First filter adds to the output of the first channel only, etc.

    // feature map (or activation map) is the output of one filter (or kernel or
    // detector)
    vector<vector<vector<double>>> output_block;
    for (int i = 0; i < num_filters; i++) {  // Should be embarrassingly parallel
      vector<vector<double>> feature_map = convolve(a, filters[i], stride_per_filter[i]);
      output_block.push_back(feature_map);
    }
    return output_block;
  }

  // static because this is a self-contained method
  vector<vector<double>> static convolve(vector<vector<vector<double>>> a, vector<vector<double>> filter, int stride) {
    // a is num_channels x height x width
    // Reference:
    // https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera

    int depth = a.size();
    int height = a[0].size();
    int width = a[0][0].size();

    int depth_of_a = a.size();
    vector<vector<double>> feature_map = _convolve(a[0], filter, stride);
    for (int i = 1; i < depth; i++) {
      vector<vector<double>> feature_map_for_depth = _convolve(a[i], filter, stride);
      feature_map = add_matrices(feature_map, feature_map_for_depth);
    }

    for (int depth = 0; depth < depth_of_a; depth++) {
    }

    return feature_map;
  }

  // Need to take into account stride.
  vector<vector<double>> static _convolve(vector<vector<double>> a, vector<vector<double>> filter, int stride) {
    // Height and width of the convolution.
    int c_width = (a.size() - filter.size()) / stride + 1;
    int c_height = (a[0].size() - filter.size()) / stride + 1;

    vector<vector<double>> convolved(c_width, (vector<double>(c_height, 0)));
    for (int i = 0; i < c_width; ++i) {
      for (int j = 0; j < c_height; ++j) {
        for (int x = 0; x < filter.size(); ++x) {
          for (int y = 0; y < filter[0].size(); ++y) {
            convolved[i][j] = convolved[i][j] + a[i * stride + x][j * stride + y] * filter[x][y];
          }
        }
      }
    }

    return convolved;
  }

  void static _convolve_test() {
    vector<vector<double>> a = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                {0.0, 1.0, 1.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0, 1.0, 1.0},
                                {0.0, 0.0, 1.0, 1.0, 0.0},
                                {0.0, 1.0, 1.0, 0.0, 0.0}};
    vector<vector<double>> filter = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}};

    vector<vector<double>> actual_output = _convolve(a, filter, 2);
    vector<vector<double>> expected_output = {{4.0, 4.0}, {2.0, 4.0}};

    for (int i = 0; i < actual_output.size(); i++) {
      for (int j = 0; j < actual_output[i].size(); j++) {
        cout << actual_output[i][j] << ",";
        if (actual_output[i][j] != expected_output[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<double>> a2 = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                 {0.0, 1.0, 1.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0, 1.0, 1.0},
                                 {0.0, 0.0, 1.0, 1.0, 0.0},
                                 {0.0, 1.0, 1.0, 0.0, 0.0}};
    vector<vector<double>> filter2 = {{1.0}};

    vector<vector<double>> actual_output2 = _convolve(a2, filter2, 1);
    vector<vector<double>> expected_output2 = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                               {0.0, 1.0, 1.0, 1.0, 0.0},
                                               {0.0, 0.0, 1.0, 1.0, 1.0},
                                               {0.0, 0.0, 1.0, 1.0, 0.0},
                                               {0.0, 1.0, 1.0, 0.0, 0.0}};

    for (int i = 0; i < actual_output2.size(); i++) {
      for (int j = 0; j < actual_output2[i].size(); j++) {
        cout << actual_output2[i][j] << ",";
        if (actual_output2[i][j] != expected_output2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }

  void static convolve_test() {
    vector<vector<vector<double>>> a = {{{1.0, 1.0, 1.0, 0.0, 0.0},
                                         {0.0, 1.0, 1.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0, 1.0, 1.0},
                                         {0.0, 0.0, 1.0, 1.0, 0.0},
                                         {0.0, 1.0, 1.0, 0.0, 0.0}},

                                        {{0.0, 1.0, 0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0, 1.0, 1.0},
                                         {0.0, 0.0, 1.0, 1.0, 0.0},
                                         {0.0, 1.0, 1.0, 0.0, 1.0},
                                         {0.0, 1.0, 1.0, 0.0, 0.0}},

                                        {{1.0, 0.0, 0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0, 0.0, 0.0},
                                         {0.0, 0.0, 1.0, 0.0, 0.0},
                                         {0.0, 0.0, 0.0, 1.0, 0.0},
                                         {0.0, 0.0, 0.0, 0.0, 1.0}}};
    vector<vector<double>> filter = {{1, 0}, {1, 1}};

    vector<vector<double>> actual_output = convolve(a, filter, 2);
    vector<vector<double>> expected_output = {{4.0, 5.0}, {1.0, 7.0}};

    for (int i = 0; i < actual_output.size(); i++) {
      for (int j = 0; j < actual_output[i].size(); j++) {
        cout << actual_output[i][j] << ",";
        if (actual_output[i][j] != expected_output[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<vector<double>>> a2 = {{{9.0, 1.0, 9.0, 0.0, 9.0},
                                          {0.0, 1.0, 9.0, 1.0, 0.0},
                                          {9.0, 0.0, 9.0, 1.0, 9.0},
                                          {0.0, 0.0, 9.0, 0.0, 0.0},
                                          {9.0, 1.0, 9.0, 0.0, 9.0}}};
    vector<vector<double>> filter2 = {{1.0}};

    vector<vector<double>> actual_output2 = convolve(a2, filter2, 2);
    vector<vector<double>> expected_output2 = {{9.0, 9.0, 9.0}, {9.0, 9.0, 9.0}, {9.0, 9.0, 9.0}};

    for (int i = 0; i < actual_output2.size(); i++) {
      for (int j = 0; j < actual_output2[0].size(); j++) {
        cout << actual_output2[i][j] << ",";
        if (actual_output2[i][j] != expected_output2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }
};

class Pool : public Layer {};

class MaxPool : public Pool {
 public:
  int height;
  int width;
  int stride;

  // No num_input_channels variable is necessary because no weights are
  // allocated for Pooling
  MaxPool(int size) {
    this->height = size;
    this->width = size;
    this->stride = size;
  }

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    vector<vector<vector<double>>> output_block;

    int num_input_channels = a.size();

    for (int i = 0; i < num_input_channels; i++) {                               // Should be embarrassingly parallel
      vector<vector<double>> pool_map = _max_pool(a[i], height, width, stride);  // Max pool by later
      output_block.push_back(pool_map);
    }
    return output_block;
  }

  vector<vector<double>> static _max_pool(vector<vector<double>> a, int height, int width, int stride) {
    int i = 0;
    int j = 0;

    vector<vector<double>> pool_map;
    while (i <= a.size() - height) {
      vector<double> pooled_row;

      while (j <= a[0].size() - width) {
        double max_value = numeric_limits<double>::lowest();
        for (int x = 0; x < height && i + x < a.size(); ++x) {
          for (int y = 0; y < width && j + y < a[0].size(); ++y) {
            if (a[i + x][j + y] > max_value) {
              max_value = a[i + x][j + y];
            }
          }
        }
        pooled_row.push_back(max_value);
        j = j + stride;
      }
      pool_map.push_back(pooled_row);
      j = 0;
      i = i + stride;
    }
    return pool_map;
  }

  void static _max_pool_test() {
    vector<vector<double>> a = {{0, 1, 2, 3}, {4, 5, 6, 7}, {1, 1, 1, 1}, {9, 0, 6, 3}};

    vector<vector<double>> test_val = _max_pool(a, 2, 2, 2);
    vector<vector<double>> expected_val = {{5, 7}, {9, 6}};
    for (int i = 0; i < test_val.size(); ++i) {
      for (int j = 0; j < test_val[0].size(); ++j) {
        cout << test_val[i][j] << ",";
        if (test_val[i][j] != expected_val[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<double>> a2 = {{0, 1, 2, 3}, {4, 5, 6, 7}, {1, 1, 1, 1}, {9, 0, 6, 3}};

    vector<vector<double>> test_val2 = _max_pool(a2, 1, 2, 3);
    vector<vector<double>> expected_val2 = {{1}, {9}};
    for (int i = 0; i < test_val2.size(); ++i) {
      for (int j = 0; j < test_val2[0].size(); ++j) {
        cout << test_val2[i][j] << ",";
        if (test_val2[i][j] != expected_val2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }
};

class Act : public Layer {
 public:
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> z) {
    // Applied the sigmoid element wise.
    vector<vector<vector<double>>> output_block;
    for (int i = 0; i < z.size(); i++) {
      vector<vector<double>> row_output;
      for (int j = 0; j < z[0].size(); j++) {
        vector<double> depth_output;
        for (int k = 0; k < z[0][0].size(); k++) {
          double activation = activation_func(z[i][j][k]);
          depth_output.push_back(activation);
        }
        row_output.push_back(depth_output);
      }
      output_block.push_back(row_output);
    }
    return output_block;
  }

  vector<double> h(vector<double> z) {
    // Applied the sigmoid element wise.
    vector<double> output_vector;
    for (int i = 0; i < z.size(); i++) {
      double activation = activation_func(z[i]);
      output_vector.push_back(activation);
    }
    return output_vector;
  }

  virtual double activation_func(double z) = 0;
};

class Sigmoid : public Act {
 public:
  double activation_func(double z) { return 1 / (1 + exp(-z)); }

  void static sigmoid_test() {
    vector<vector<vector<double>>> z = {{{1, 1}, {2, 2}, {3, 3}}, {{1, 0}, {0, 1}, {1, -1}}};
    vector<vector<vector<double>>> val = Sigmoid().h(z);
    vector<vector<vector<double>>> expected = {{{0.731059, 0.731059}, {0.880797, 0.880797}, {0.952574, 0.952574}},
                                               {{0.731059, 0.5}, {0.5, 0.731059}, {0.731059, 0.268941}}};

    for (int i = 0; i < z.size(); i++) {
      for (int j = 0; j < z[0].size(); j++) {
        for (int k = 0; k < z[0][0].size(); k++) {
          cout << val[i][j][k] << ", ";
          if (abs(val[i][j][k] - expected[i][j][k]) > 0.0001) {
            throw(string) "Test failed! " + (string) __FUNCTION__;
          }
        }
      }
    }

    cout << endl;
  }
};

class Relu : public Act {
 public:
  double activation_func(double z) { return max(0.0, z); }

  void static relu_test() {
    vector<vector<vector<double>>> z = {{{1, 1}, {2, -2}, {3, -3}}, {{1, 0}, {0, 1}, {1, -1}}};
    vector<vector<vector<double>>> val = Relu().h(z);
    vector<vector<vector<double>>> expected = {{{1, 1}, {2, 0}, {3, 0}}, {{1, 0}, {0, 1}, {1, 0}}};

    for (int i = 0; i < z.size(); i++) {
      for (int j = 0; j < z[0].size(); j++) {
        for (int k = 0; k < z[0][0].size(); k++) {
          cout << val[i][j][k] << ", ";
          if (val[i][j][k] != expected[i][j][k]) {
            throw(string) "Test failed! " + (string) __FUNCTION__;
          }
        }
      }
    }

    cout << endl;
  }
};

class Flatten : public Layer {
  // Flattens to a column vector
 public:
  vector<double> static f(vector<vector<vector<double>>> a) {
    vector<double> flattened;
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        for (int k = 0; k < a[0][0].size(); k++) {
          flattened.push_back(a[i][j][k]);  // Add a one element row vector to the column
        }
      }
    }
    return flattened;
  }
};

class Dense : public Layer {
 public:
  int num_in;
  int num_out;

  vector<vector<double>> weights;
  vector<double> biases;

  Dense(int num_in, int num_out) {
    this->num_in = num_in;
    this->num_out = num_out;

    // Initialize weights with all values zero, then set all weights to a random value
    weights = vector<vector<double>>(num_in, vector<double>(num_out, 0));
    rand_init(weights, num_in, num_out);

    // Initialize biases with all values zero, then set all biases to a random value
    biases = vector<double>(num_out, 0);
    rand_init(biases, num_out);
  }

  vector<double> h(vector<double> a) {
    vector<double> zs;

    if (a.size() != num_in) {
      throw(string) "Mismatch between Dense parameters and incoming vector!";
    }

    for (int i = 0; i < num_out; i++) {
      double z = biases[i];
      for (int j = 0; j < num_in; j++) {
        z = z + weights[i][j] * a[j];
      }
      zs.push_back(z);
    }

    return zs;
  }

  void static h_test() {
    vector<double> a{1, 2, 3};

    Dense d = Dense(3, 5);
    d.weights = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}};
    d.biases = {0, 0, 0};

    vector<double> output = d.h(a);
    vector<double> expected_output = {1, 2, 3, 0, 0, 0};
    for (int i = 0; i < output.size(); i++) {
      if (output[i] != expected_output[i]) {
        throw(string) "Test failed! " + (string) __FUNCTION__;
      }
    }

    vector<double> a2{1, 2, 3};

    Dense d2 = Dense(3, 5);
    d2.weights = {{1, 1, 0}, {0, 1, 3}, {0, 0, 1}, {1, 0, 0}, {0, 2, 0}};
    d2.biases = {0, 0, 0};

    vector<double> output2 = d2.h(a2);
    vector<double> expected_output2 = {3, 11, 3, 1, 4, 0};
    for (int i = 0; i < output2.size(); i++) {
      if (output2[i] != expected_output2[i]) {
        throw(string) "Test failed! " + (string) __FUNCTION__;
      }
    }
  }
};

class ConvNet {
 public:
  vector<Layer*> layers;
  ConvNet(vector<Layer*> layers) { this->layers = layers; }

  int h(vector<vector<vector<double>>> x) {  // Returns an int, a classification
    vector<vector<vector<double>>> a = x;
    vector<double> v;

    for (Layer* layer : layers) {
      bool is_last_output_vector = false;
      if (Conv* conv = dynamic_cast<Conv*>(layer)) {
        a = conv->h(a);
      } else if (MaxPool* pool = dynamic_cast<MaxPool*>(layer)) {
        a = pool->h(a);
      } else if (Act* act = dynamic_cast<Act*>(layer)) {
        if (is_last_output_vector) {
          v = act->h(v);
        } else {
          a = act->h(a);
        }
      } else if (Flatten* flatten = dynamic_cast<Flatten*>(layer)) {
        v = flatten->f(a);
        is_last_output_vector = true;
      } else if (Dense* dense = dynamic_cast<Dense*>(layer)) {
        v = dense->h(v);
        is_last_output_vector = true;
      }
    }

    // Take argmax of the output
    int label = 0;
    cout << v[0] << ",";
    for (int i = 1; i < v.size(); i++) {
      cout << v[i] << ",";
      if (v[label] < v[i]) {
        label = i;
      }
    }
    cout << endl;

    return label;
  }

  void static h_test(vector<vector<vector<vector<double>>>> X) {
    // Intialize model and evaluate an example test
    // Compound literal, (vector[]), helps initialize an array in function call
    Conv conv = Conv(1, 2, (vector<int>){3, 3}, (vector<int>){1, 1});
    MaxPool pool = MaxPool(2);
    Relu relu = Relu();
    Flatten flatten = Flatten();
    Dense dense = Dense(338, 10);
    Sigmoid sigmoid = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&conv, &pool, &relu, &flatten, &dense, &sigmoid});
    // Do a forward pass with the first "image"
    int label = model.h(X[0]);
    cout << label << endl;

    if (!(label >= 0 && 10 > label)) {
      throw(string) "Test failed! " + (string) __FUNCTION__;
    }
  }
};

int main() {
  // TESTS
  // set up

  const int num_images = 100;
  vector<vector<vector<vector<double>>>> X;  // num_images x num_channels x height x width
  int Y[num_images];                         // labels for each example

  // Randomly initialize X and Y
  for (int i = 0; i < num_images; i++) {
    vector<vector<vector<double>>> image;
    vector<vector<double>> channel;  // Only one channel per image here.
    for (int j = 0; j < 28; j++) {
      vector<double> row;
      for (int k = 0; k < 28; k++) {
        double f = (double)rand() / RAND_MAX;
        double num = {255 * f};  // use numbers from 0 to 255
        row.push_back(num);
      }
      channel.push_back(row);
    }
    image.push_back(channel);
    X.push_back(image);
    Y[i] = rand() % 10;  // TODO: Maybe decrease number of classes for the test?
  }

  // Look at first 2 "images"
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        cout << X[i][0][j][k] << ",";
      }
      cout << endl;
    }
    cout << endl;
  }

  // tests

  try {
    // Flat convolution test
    Conv::_convolve_test();
    cout << "_convole_test done" << endl;

    // Depth convolution test
    Conv::convolve_test();
    cout << "convole_test done" << endl;

    // Flat max pool test
    MaxPool::_max_pool_test();
    cout << "_max_pool_test done" << endl;

    // TODO: make a depth maxpool test if necessary

    Sigmoid::sigmoid_test();
    cout << "sigmoid_test done" << endl;

    Relu::relu_test();
    cout << "relu_test done" << endl;

    Dense::h_test();
    cout << "Dense h_test done" << endl;

    ConvNet::h_test(X);
    cout << "ConvNet h_test done" << endl;

  } catch (string my_exception) {
    cout << my_exception << endl;
    return 0;  // Do not go past the first exception in a test
  }

  cout << "Tests finished!!!!\n";

  // Main

  return 0;
}
