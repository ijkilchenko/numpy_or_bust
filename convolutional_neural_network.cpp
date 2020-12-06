#include <math.h>

#include <iostream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

// Good diagram:
// https://engmrk.com/wp-content/uploads/2018/09/Image-Architecture-of-Convolutional-Neural-Network.png

class Layer {
 public:
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> x);

  // Helper functions
  static void rand_init(vector<vector<double>> &matrix, int height, int width) {
    for (int i = 0; i < height; i++) {
      vector<double> row;
      for (int j = 0; j < width; j++) {
        // use numbers between -100 and 100
        double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
        n = n * 200 - 100;
        matrix[i][j] = n;  // (possibly) change to use float to save memory
      }
    }
  }

  static void rand_init(vector<double> matrix, int length) {
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
    num_input_channels = num_input_channels;
    num_filters = num_filters;
    size_per_filter = size_per_filter;
    stride_per_filter = stride_per_filter;

    for (int filter_num; filter_num < num_filters; filter_num++) {
      // Filters are square
      int height = size_per_filter[filter_num];
      int width = size_per_filter[filter_num];

      vector<vector<double>> filter;
      Layer().rand_init(filter, height, width);
      filters[filter_num] = filter;
    }
  }

  // TODO: Write a test for this function if needed.
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    // Input and output is height x width x num_channels
    // First filter adds to the output of the first channel only, etc.

    // feature map (or activation map) is the output of one filter (or kernel or
    // detector)
    vector<vector<vector<double>>> output_block;
    int num_filters = filters.size();
    for (int i = 0; i < num_filters; i++) {  // Should be embarrassingly parallel
      vector<vector<double>> feature_map = convolve(a, filters[i], stride_per_filter[i]);
      output_block.push_back(feature_map);
    }
    return output_block;
  }

  // static because this is a self-contained method
  vector<vector<double>> static convolve(vector<vector<vector<double>>> a, vector<vector<double>> filter, int stride) {
    // a is height x width x num_channels
    // Let's say a is 10x10x3 and filter is 3x3
    // The first convolutional step will use a's top left corner block of size
    // 3x3x3 For each (i, j, [1, 2, 3]) section of a, we use the same (i, j)th
    // weight of filter to flatten it In other words, we do a[i][j][1]*w +
    // a[i][j][2]*w + a[i][j][3]*w. This produces a 3x3x1 which we then
    // element-wise multiply with filter which is also 3x3x1 to produce 3x3x1
    // multiplications. These are then all added together to produce one output
    // per convolutional step. Reference:
    //
    // https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera

    int height = a.size();
    int width = a[0].size();
    int depth = a[0][0].size();

    int depth_of_a = a.size();
    vector<vector<double>> feature_map = _convolve(a[0], filter, stride);
    for (int depth = 1; depth < depth_of_a; depth++) {
      vector<vector<double>> feature_map_for_depth = _convolve(a[depth], filter, stride);
      feature_map = add_matrices(feature_map, feature_map_for_depth);
    }

    for(int depth = 0; depth < depth_of_a; depth++){

    }

    //vector<vector<double>> feature_map = _convolve(a[0], filter, stride);




    return feature_map;
  }

  // vector<vector<double>> static _convolve(vector<vector<double>> a, vector<vector<double>> filter, int stride) {
  //   // Row major seems faster
  //   // https://stackoverflow.com/questions/33722520/why-is-iterating-2d-array-row-major-faster-than-column-major
  //   int i = 0;
  //   int j = 0;

  //   vector<vector<double>> convolved;
  //   while (i <= a.size() - filter.size()) {
  //     vector<double> convolved_row;

  //     while (j <= a[0].size() - filter[0].size()) {
  //       double acc_sum{0.0};
  //       for (int x = 0; x < filter.size(); ++x) {
  //         for (int y = 0; y < filter[0].size(); ++y) {
  //           acc_sum = acc_sum + a[i + x][j + y] * filter[x][y];
  //         }
  //       }
  //       convolved_row.push_back(acc_sum);

  //       j = j + stride;
  //     }
  //     convolved.push_back(convolved_row);
  //     j = 0;

  //     i = i + stride;
  //   }
  //   return convolved;
  // }

  //Need to take into account stride.
  vector<vector<double>> static _convolve(vector<vector<double>> a, vector<vector<double>> filter, int stride){

    //Height and width of the convolution. 
    int c_width = (a.size()-filter.size())/stride + 1;
    int c_height = (a[0].size()-filter.size())/stride + 1; 

    vector<vector<double>> convolved(c_width, (vector<double>(c_height, 0)));
    for(int i = 0; i < c_width; ++i){
      for(int j = 0; j < c_height; ++j){
      
        for(int x = 0; x < filter.size(); ++x){
          for(int y = 0; y < filter[0].size(); ++y){
            convolved[i][j] = convolved[i][j]+a[i*stride+x][j*stride+y]*filter[x][y];
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
    height = size;
    width = size;
    stride = size;
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
        ;
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
  // Flattens to a row vector
 public:
  vector<double> h(vector<vector<vector<double>>> a) {
    vector<double> flattened;
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        for (int k = 0; k < a[0][0].size(); k++) {
          flattened.push_back(a[i][j][k]);  // Add a one element vector to the row
        }
      }
    }
    return flattened;
  }

  // Uncomment if we need a column vector instead
  // // Flattens to a column vector
  // public:
  //   vector<vector<double>> h(vector<vector<vector<double>>> a) {
  //     vector<vector<double>> flattened;
  //     for (int i = 0; i < a.size(); i ++) {
  //       for (int j = 0; j < a[0].size(); j++) {
  //         for (int k = 0; k < a[0][0].size(); k++) {
  //           flattened.push_back(vector<double>{a[i][j][k]}); // Add a one
  //           element vector to the column
  //         }
  //       }
  //     }
  //     return flattened;
  //   }
};

class Dense : public Layer {
 public:
  int num_in;
  int num_out;

  vector<vector<double>> weights;
  vector<double> biases;

  Dense(int num_in, int num_out) {
    // num_in = num_in;
    // num_out = num_out;

    // Initialize weights with all values zero, then set all weights to a random value
    weights = vector<vector<double>>(num_in, vector<double>(num_out, 0));
    rand_init(weights, num_in, num_out);

    // Initialize biases with all values zero, then set all biases to a random value
    biases = vector<double>(num_out, 0);
    rand_init(biases, num_out);
    cout << "rand_init #2 \n";
  }

  // Possible problems:
  // Dense's and Flatten's have a different function signature (every other
  // layer takes in a block and outputs a block) How to reuse an activation
  // function from Act layer in Dense?
  vector<double> h(vector<double> a) {
    vector<double> zs;

    if (a.size() != num_in) {
      cout << a.size() << " " << num_in << endl;
      throw(string) "Mismatch between Dense parameters and incoming vector!";
    }

    for (int i = 0; i < num_out; i++) {
      double z = biases[i];
      for (int j = 0; j < num_in; j++) {
        z = z + a[j] * weights[j][i];
      }
      zs.push_back(z);
    }

    return zs;
  }

  void static h_test() {
    vector<double> a{1, 2, 3, 4, 5};

    Dense d = Dense(d.num_in = 5, d.num_out = 3);
    d.weights = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}};
    d.biases = {0, 0, 0};

    vector<double> output = d.h(a);
    for (auto i : output) {
      cout << i << endl;
    }
  }
};

class ConvNet {
 public:
  vector<Layer> layers;
  ConvNet(vector<Layer> layers) { layers = layers; }

  // int h(vector<vector<vector<double>>> x) {  // Returns an int, a
  // classification
  //   vector<vector<vector<double>>> a = x;
  //   for (Layer layer : layers) {
  //     vector<vector<vector<double>>> a = layer.h(a);
  //   }
  //   // Convert the final output into a classification
  // }
};

int main() {
  // TESTS
  // set up

  const int num_images = 100;
  vector<vector<vector<vector<double>>>> X;  // num_images x height x width x num_channels
  int Y[num_images];                         // labels for each example

  // Randomly initialize X and Y
  for (int i = 0; i < num_images; i++) {
    vector<vector<vector<double>>> image;
    for (int j = 0; j < 28; j++) {
      vector<vector<double>> row;  // Row has depth (of 1 in this example)
      for (int k = 0; k < 28; k++) {
        double f = (double)rand() / RAND_MAX;
        vector<double> num = {255 * f};  // use numbers from 0 to 255
        row.push_back(num);
      }
      image.push_back(row);
    }
    X.push_back(image);
    Y[i] = rand() % 10;  // TODO: Maybe decrease number of classes for the test?
  }

  // Look at first 2 "images"
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        cout << X[i][j][k][0] << ",";
      }
      cout << endl;
    }
    cout << endl;
  }

  // tests

  try {

    // Flat convolution test
    Conv::_convolve_test();

    // Depth convolution test
    Conv::convolve_test();

    // Flat max pool test
    MaxPool::_max_pool_test();

    // TODO: make a depth maxpool test if necessary

    Sigmoid::sigmoid_test();

    Relu::relu_test();

    Dense::h_test();

  } catch (string my_exception) {
    cout << my_exception << endl;
    return 0;  // Do not go past the first exception in a test
  }

  // Intialize model
  // Compound literal, (vector[]), helps initialize an array in function call
  // ConvNet model = ConvNet(vector<Layer>{Conv(1, 4, (vector<int>){3, 3, 5, 5},
  // (vector<int>){1, 1, 2, 2})});

  // Do a forward pass with the first "image"
  // model.h(X[1]);

  cout << "Tests finished!!!!\n";

  // Main

  return 0;
}
