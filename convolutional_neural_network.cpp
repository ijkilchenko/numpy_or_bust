#include <iostream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

int find_int_in_str(string str) {
  string output = std::regex_replace(str, std::regex("[^0-9]*([0-9]+).*"),
                                     std::string("$1"));
  int id = stoi(output);
  return id;
};

class Layer {
 public:
  // static void rand_init(double **matrix, int rows, int cols) {
  //   for (int i = 0; i < rows; i++) {
  //     for (int j = 0; j < cols; j++) {
  //       //use numbers between -100 and 100
  //       double n = (double)rand() / RAND_MAX;
  //       n = -100 + n * 200;
  //       matrix[i][j] = n;
  //     }
  //   }
  // }

  template <size_t rows, size_t cols>
  static void rand_init(int (&matrix)[rows][cols]) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        //use numbers from 0 to 255
        matrix[i][j] = rand() % 255;
      }
    }
  }

};

class Conv : public Layer {
 public:
  int num_input_channels;
  int num_filters;
  int size_per_filter[];
  int stride_per_filter[];

  // int filters[][][];
  Conv(int num_input_channels, int num_filters, int size_per_filter[],
       int stride_per_filter[]) {
    num_input_channels = num_input_channels;
    num_filters = num_filters;
    size_per_filter = size_per_filter;
    stride_per_filter = stride_per_filter;

    for (int filter_num; filter_num < num_filters; filter_num++) {
      int width = size_per_filter[filter_num];
      int height = size_per_filter[filter_num];

      double filter[width][height];
    }
  }
};

class Pool : public Layer {};

class Act : public Layer {};

class Dense : public Layer {};

class ConvNet {
 public:
  ConvNet(vector<Layer> layers) {
    vector<Layer> layers;  // vector is a variable length array
  }
};

int main() {
  // TEST
  cout << "Starting test...\n";

  string str = "hello123world";
  int found_int = find_int_in_str(str);
  if (found_int != 123) {
    throw;
  }

  int num_images = 100;
  int X[num_images][28][28];
  int Y[num_images];

  // Randomly initialize X and Y
  for (int i = 0; i < num_images; i++) {
    Layer().rand_init(X[i]);
    Y[i] = rand() % 10;  // TODO: Maybe decrease number of classes for the test?
  }

  // Look at first 2 "images"
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        cout << X[j][k][i] << ",";
      }
      cout << endl;
    }
    cout << endl;
  }

  // Compound literal, (int[]), helps initialize an array in function call
  Conv(4, (int[]){3, 5}, (int[]){1, 2});

  // ConvNet model(vector<Layer>{Conv(4, [3, 5], [1, 2]), Pool("max"),
  // Act("relu"), Dense());
  // ConvNet model(Layer(), Layer(), Layer());

  cout << "Test finished!\n";

  // Main

  return 0;
}
