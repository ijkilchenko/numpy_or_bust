#include <iostream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

// Good diagram: https://engmrk.com/wp-content/uploads/2018/09/Image-Architecture-of-Convolutional-Neural-Network.png

class Layer {
 public:
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> x);

  // Helper function
  static void rand_init(vector<vector<double>> matrix, int height, int width) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // use numbers between -100 and 100
        double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
        n = n * 200 - 100;
        matrix[i][j] = n;  // (possibly) change to use float to save memory
      }
    }
  }
};

class Conv : public Layer {
 public:
  int num_input_channels;
  int num_filters;
  vector<int> size_per_filter;
  vector<int> stride_per_filter;

  vector<vector<vector<double>>> filters;
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

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    // Input and output is height x width x num_channels
    // First filter adds to the output of the first channel only, etc.

    // feature map (or activation map) is the output of one filter (or kernel or detector)
    vector<vector<vector<double>>> output_block;
    int num_filters = filters.size();
    for (int i = 0; i < num_filters; i++) {  // Should be embarrassingly parallel
      vector<vector<double>> feature_map = convolve(a, filters[i], stride_per_filter[i]);
    }
  }

  // static because this is a self-contained method
  vector<vector<double>> static convolve(vector<vector<vector<double>>> a, vector<vector<double>> filter, int stride) {
    // a is height x width x num_channels
    // Let's say a is 10x10x3 and filter is 3x3
    // The first convolutional step will use a's top left corner block of size 3x3x3
    // For each (i, j, [1, 2, 3]) section of a, we use the same (i, j)th weight of filter to flatten it
    // In other words, we do a[i][j][1]*w + a[i][j][2]*w + a[i][j][3]*w.
    // This produces a 3x3x1 which we then element-wise multiply with filter which is also 3x3x1 to
    // produce 3x3x1 multiplications. These are then all added together to produce one output per
    // convolutional step.
    // Reference:
    // https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera

    int height = a.size();
    int width = a[0].size();
    int depth = a[0][0].size();

    vector<vector<double>> feature_map;
    int depth_of_a = a.size();
    for (int depth = 0; depth < depth_of_a; depth++) {
      vector<vector<double>> feature_map_per_depth = _convole(a[depth], filter, stride);
    }

    // TODO
  }

  vector<vector<double>> static _convole(vector<vector<double>> a, vector<vector<double>> filter, int stride) {
    // TODO
  }

  vector<vector<double>> static _convole_test() {
    
  }
};

class Pool : public Layer {};

class Act : public Layer {};

class Dense : public Layer {};

class ConvNet {
 public:
  vector<Layer> layers;
  ConvNet(vector<Layer> layers) { layers = layers; }

  int h(vector<vector<vector<double>>> x) {  // Returns an int, a classification
    vector<vector<vector<double>>> a = x;
    for (Layer layer : layers) {
      vector<vector<vector<double>>> a = layer.h(a);
    }
    // Convert the final output into a classification
  }
};

int main() {
  // TEST
  cout << "Starting test...\n";

  int num_images = 100;
  vector<vector<vector<vector<double>>>> X;  // num_images x height x width x num_channels
  int Y[num_images];                         // labels for each example

  // Randomly initialize X and Y
  for (int i = 0; i < num_images; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        // use numbers from 0 to 255
        X[i][j][k][1] = rand() % 255;
      }
    }
    Y[i] = rand() % 10;  // TODO: Maybe decrease number of classes for the test?
  }

  // Look at first 2 "images"
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        cout << X[j][k][i][1] << ",";
      }
      cout << endl;
    }
    cout << endl;
  }

  // Intialize model
  // Compound literal, (vector[]), helps initialize an array in function call
  ConvNet model = ConvNet(vector<Layer>{Conv(1, 4, (vector<int>){3, 3, 5, 5}, (vector<int>){1, 1, 2, 2})});

  // Do a forward pass with the first "image"
  model.h(X[1]);

  cout << "Test finished!\n";

  // Main

  return 0;
}
