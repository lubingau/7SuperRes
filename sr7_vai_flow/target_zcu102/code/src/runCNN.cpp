#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

void interpolateImages(const std::vector<cv::Mat>& inputImages, std::vector<cv::Mat>& outputImages) {
    cout << "[SR7 INFO RunCNN] Interpolating images..." << endl;
    for (const auto& inputImage : inputImages) {
        cv::Mat outputImage;
        cv::resize(inputImage, outputImage, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        outputImages.push_back(outputImage);
    }
    cout << "[SR7 INFO RunCNN] Interpolation done!" << endl;
}

GraphInfo shapes;

// const string baseImagePath = "./src/img_test/";
string baseImagePath;  // they will get their values via argv[]

int num_threads = 0;
int is_running_0 = 1;
int num_of_images = 0;
int num_images_x_thread = 0;


// void ListImages(string const &path, vector<string> &images_list) {
//   images_list.clear();
//   struct dirent *entry;

//   /*Check if path is a valid directory path. */
//   struct stat s;
//   lstat(path.c_str(), &s);
//   if (!S_ISDIR(s.st_mode)) {
//     fprintf(stderr, "[SR7 ERROR] %s is not a valid directory!\n", path.c_str());
//     exit(1);
//   }

//   DIR *dir = opendir(path.c_str());
//   if (dir == nullptr) {
//     fprintf(stderr, "[SR7 ERROR] Open %s path failed.\n", path.c_str());
//     exit(1);
//   }

//   while ((entry = readdir(dir)) != nullptr) {
//     if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
//       string name = entry->d_name;
//       string ext = name.substr(name.find_last_of(".") + 1);
//       if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
//           (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
//         images_list.push_back(name);
//       }
//     }
//   }

//   closedir(dir);
// }

void runDPU(vart::Runner *runner, uint8_t *inputBuffer, uint8_t *outputBuffer)
{
  // get in/out tensors and dims
  auto outputTensors = runner->get_output_tensors();
  // size of the output tensor
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();
  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  uint8_t *loc_inputBuffer = inputBuffer;
  uint8_t *loc_outputBuffer = outputBuffer;

  cout << "[SR7 INFO] Inside RUN CNN " << endl;

  for (unsigned int n = 0; n < num_images_x_thread; n += batchSize)  // this works correctly for either batchSize= 1 or 3
  {
    loc_inputBuffer = inputBuffer + n * inSize;
    loc_outputBuffer = outputBuffer + n * outSize;

    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(loc_inputBuffer, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>( loc_outputBuffer, batchTensors.back().get()));

    // tensor buffer input/output
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    // run
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

    inputs.clear();
    outputs.clear();
  }
}

/**
 * @brief Entry for running FCN8 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy FCN8 on DPU platform.
 *
 */


void runCNN(vector<Mat>& inputImages, vector<Mat>& outputImages, string xmodel_path, int num_threads) {

  // Check args
  // if (argc != 7) {
  //   cout << "Usage: run_cnn xmodel_path test_images_path thread_num (from 1 to "
  //           "6) use_post_proc(1:yes, 0:no) save_results(1:yes, 0:no) num_of_images"
  //        << endl;
  //   return -1;
  // }
  // baseImagePath =
  //     std::string(argv[2]);  // path name of the folder with test images
  // num_threads = atoi(argv[3]);
  assert((num_threads <= 6) & (num_threads >= 1));
  // int use_post_processing = atoi(argv[4]);
  // int save_results = atoi(argv[5]);
  // int NUM_TEST_IMAGES = atoi(argv[6]);

  // for (int i=0; i<argc; i++) cout << argv[i] << " "; cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPARE DPU STUFF

  auto graph = xir::Graph::deserialize(xmodel_path);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "[SR7 INFO] Create running for subgraph: " << subgraph[0]->get_name();
  cout << "[SR7 INFO] Create running for subgraph: " << subgraph[0]->get_name() << endl;

  // create runners
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  // get in/out tensors and dims
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  // deprecated
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  // get in/out tensor shape
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;

  int out_fixpos = (outputTensors[0])->template get_attr<int>("fix_point");
  auto out_fix_scale = std::exp2f(1.0f * (float)out_fixpos);
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int in_fixpos = (inputTensors[0])->template get_attr<int>("fix_point");
  auto in_fix_scale = std::exp2f(1.0f * (float)in_fixpos);
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);

  float input_scale = get_input_scale(inputTensors[0]);
  float output_scale = get_output_scale(outputTensors[0]);

  // debug messages
  cout << "outSize  " << outSize << endl;
  cout << "inSize   " << inSize << endl;
  cout << "outW     " << outWidth << endl;
  cout << "outH     " << outHeight << endl;
  cout << "inpW     " << inWidth << endl;
  cout << "inpH     " << inHeight << endl;
  cout << "# class  " << num_of_classes << endl;
  cout << "batchSize " << batchSize << endl;  // alway 1 for Edge

  cout << "in_fixpos     " << in_fixpos << endl;
  cout << "in_fix_scale  " << in_fix_scale << endl;
  cout << "inputt_scale  " << input_scale << endl;
  cout << "out fix scale " << out_fix_scale << endl;
  cout << "output_scale "  << output_scale << endl;


  /////////////////////////////////////////////////////////////////////////////////////////////
  // TIMERS CALIBRATION

  int num_of_trials = 200;
  std::chrono::duration<double, std::micro> avg_calibr_highres(0);
  for (int i =0; i<num_of_trials; i++)
  {
      auto t1 = std::chrono::high_resolution_clock::now();
      auto t2 = std::chrono::high_resolution_clock::now();
      // floating-point duration: no duration_cast needed
      std::chrono::duration<double, std::micro> fp_us = t2 - t1;
      avg_calibr_highres  += fp_us;
      //if (i%20 ==0) cout << "[Timers calibration  ] " << fp_us.count() << "us" << endl;
    }
  cout << "[average calibration high resolution clock] " << avg_calibr_highres.count() / num_of_trials << "us"  << endl;
  cout << "\n" << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY ALLOCATION

  // // Load all image filenames
  // vector<string> IMAGES_NAME_LIST;
  // ListImages(baseImagePath, IMAGES_NAME_LIST);
  // //std::sort(IMAGES_NAME_LIST.begin(), IMAGES_NAME_LIST.end());

  // if (IMAGES_NAME_LIST.size() == 0) {
  //   cerr << "\n[SR7 ERROR] No images existing under " << baseImagePath << endl;
  //   exit(-1);
  // } else {
  //   num_of_images = IMAGES_NAME_LIST.size();
  //   cout << "\n[SR7 INFO] Find " << num_of_images << " images" << endl;
  // }

  // if (num_of_images > NUM_TEST_IMAGES) num_of_images = NUM_TEST_IMAGES;
  num_of_images = inputImages.size();
  cout << "\n[SR7 INFO] Number of images to read " << num_of_images << endl;

  // number of images per thread
  num_images_x_thread = num_of_images / num_threads;
  //num_images_x_thread = (num_images_x_thread / batchSize) * batchSize;
  cout << "\n[SR7 INFO] Number of images per thread: " << num_images_x_thread << endl;
  // effective number of images as multiple of num_threads and batchSize
  num_of_images = num_images_x_thread * num_threads;

  // memory allocation
  //vector<Mat> IMAGES_LIST;
  Mat image = cv::Mat(inHeight, inWidth, CV_8UC3);
  
  uint8_t *inputBuffer = new uint8_t[(num_of_images)*inSize*2];
  uint8_t *outputBuffer    = new uint8_t[(num_of_images)*outSize*2];


  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPROCESSING ALL IMAGES AT ONCE
  cout << "\n[SR7 INFO] DOING PRE PROCESSING\n" << endl;
  auto pre_t1 = std::chrono::high_resolution_clock::now();

  // for (unsigned int n = 0; n < num_of_images; n++)
  // {
  //     image = imread(baseImagePath + IMAGES_NAME_LIST[n]);
  //     // cout << "Reading " << IMAGES_NAME_LIST[n] << endl;
  //     IMAGES_LIST.push_back(image);
  // }
  
  cout << "\n[SR7 INFO] Images loaded" << endl;
  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = inputImages[n];

      for (int y = 0; y < inHeight; y++) {
	       for (int x = 0; x < inWidth; x++) {
            for (int c = 0; c < 3; c++) {
              float tmp_pix = ((float) image.at<Vec3b>(y,x)[c])/255;
              tmp_pix = tmp_pix * input_scale;
              inputBuffer[n*inSize + 3*(y*inWidth+x) + c  ] = (uint8_t) tmp_pix; //BGR format
              //inputBuffer[n*inSize + 3*(y*inWidth+x) + 2-c] = (uint8_t) tmp_pix; //RGB format
            }
	       }
      }
  }
  cout << "\n[SR7 INFO] Images loaded in the buffer" << endl;

  auto pre_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_time = pre_t2 - pre_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[PREPROC Time ] " << prepr_time.count() << "us" << endl;
  cout << "[PREPROC FPS  ] " << num_of_images*1000000.0/prepr_time.count()  << endl;
  cout << "\n" << endl;


  // split images in chunks, each chunks for its own thead
  // avoid pointing to wrong memorycv::Mat> locations
  uint8_t *inputBuffer0 =
      inputBuffer + inSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  uint8_t *inputBuffer1 =
      inputBuffer + inSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  uint8_t *inputBuffer2 =
      inputBuffer + inSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  uint8_t *inputBuffer3 =
      inputBuffer + inSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  uint8_t *inputBuffer4 =
      inputBuffer + inSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  uint8_t *inputBuffer5 =
      inputBuffer + inSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);

  uint8_t *outputBuffer0 =
      outputBuffer + outSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  uint8_t *outputBuffer1 =
      outputBuffer + outSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  uint8_t *outputBuffer2 =
      outputBuffer + outSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  uint8_t *outputBuffer3 =
      outputBuffer + outSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  uint8_t *outputBuffer4 =
      outputBuffer + outSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  uint8_t *outputBuffer5 =
      outputBuffer + outSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // MULTITHREADING DPU EXECUTION WITH BATCH
  thread workers[num_threads];

  auto dpu_t1 = std::chrono::high_resolution_clock::now();

  for (auto i = 0; i < num_threads; i++) {
    if (i == 0)
      workers[i] =
          thread(runDPU, runner.get(), ref(inputBuffer0), ref(outputBuffer0));
    if (i == 1)
      workers[i] =
          thread(runDPU, runner1.get(), ref(inputBuffer1), ref(outputBuffer1));
    if (i == 2)
      workers[i] =
          thread(runDPU, runner2.get(), ref(inputBuffer2), ref(outputBuffer2));
    if (i == 3)
      workers[i] =
          thread(runDPU, runner3.get(), ref(inputBuffer3), ref(outputBuffer3));
    if (i == 4)
      workers[i] =
          thread(runDPU, runner4.get(), ref(inputBuffer4), ref(outputBuffer4));
    if (i == 5)
      workers[i] =
          thread(runDPU, runner5.get(), ref(inputBuffer5), ref(outputBuffer5));
  }
  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  auto dpu_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> dpu_time = dpu_t2 - dpu_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[DPU tot Time ] " << dpu_time.count()                 << "us" << endl;
  //cout << "[DPU avg Time ] " << (dpu_time.count()/num_of_images) << "us" << endl;
  cout << "[DPU avg FPS  ] " << num_of_images*1000000.0/dpu_time.count()  << endl;
  cout << "\n" << endl;

  double total_time = 0.0;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // READ OUTPUT BUFFER
  cout << "\n[SR7 INFO] Read output buffer\n" << endl;

  // Create a Mat to hold the super-resolved image
  cv::Mat superResolvedImage = cv::Mat::zeros(outHeight, outWidth, CV_8UC3);
  for (unsigned int n = 0; n < num_of_images; n++) {
    uint8_t *OutData = &outputBuffer[n * outSize];
    // Iterate over rows and columns of the super-resolved image
    float tmp_pix;
    int B_pix, G_pix, R_pix;
    for (int row = 0; row < outHeight; row++) {
        for (int col = 0; col < outWidth; col++) {
            B_pix = ((float) OutData[n*outSize + 3*(row*outWidth+col) + 0]);
            G_pix = ((float) OutData[n*outSize + 3*(row*outWidth+col) + 1]);
            R_pix = ((float) OutData[n*outSize + 3*(row*outWidth+col) + 2]);
            superResolvedImage.at<cv::Vec3b>(row, col) = Vec3b(B_pix, G_pix, R_pix);
        }
    }
    outputImages.push_back(superResolvedImage);
  }

  // if (save_results == 1) {
  //   cout << "\n[SR7 INFO] SAVING RESULTS\n" << endl;
  //   string image_name;
  //   int row_index, col_index;
  //   cout << "[SR7 INFO] Number of images: " << num_of_images << endl;
  //   for (unsigned int n = 0; n < num_of_images; n++) {
      
  //     image_name = IMAGES_NAME_LIST[n];
  //     //cout << "Input image name: " << image_name << endl;
  //     // remove the extension
  //     image_name = image_name.substr(0, image_name.find_last_of("."));
  //   //   // split the name with _
  //   //   vector<string> tokens;
  //   //   stringstream check1(image_name);
  //   //   string intermediate;
  //   //   while(getline(check1, intermediate, '_'))
  //   //   {
  //   //       tokens.push_back(intermediate);
  //   //   }

  //     // indexes are 2 last tokens
  //   //   row_index = stoi(tokens[tokens.size()-2]);
  //   //   col_index = stoi(tokens[tokens.size()-1]);
  //     //cout << "row index: " << row_index << " Col index: " << col_index << endl;
  //     // Assuming outputBuffer is a pointer to the int8 tensor containing super-resolution results
  //     uint8_t *OutData = &outputBuffer[n * outSize];

  //     // Create a Mat to hold the super-resolved image
  //     cv::Mat superResolvedImage = cv::Mat::zeros(outHeight, outWidth, CV_8UC3);
  //     // Iterate over rows and columns of the super-resolved image
  //     float tmp_pix;
  //     for (int row = 0; row < outHeight; row++) {
  //         for (int col = 0; col < outWidth; col++) {
  //             for (int c = 0; c < 3; c++) {
  //                 tmp_pix = ((float) OutData[n*outSize + 3*(row*outWidth+col) + c]);
  //                 superResolvedImage.at<cv::Vec3b>(row, col)[c] = tmp_pix;
  //             }
  //         }
  //     }

  //     // Optional: Display or save the super-resolved image
  //     // if (n <= 3) {
  //     //   cv::imshow(format("super_resolved_%03d.png", n), superResolvedImage);
  //     //   cv::waitKey(1000);
  //     // }

  //     // Optional: Save the super-resolved image into the folder named "outputs"
  //     cv::imwrite(cv::format("outputs/%s.png", image_name.c_str()), superResolvedImage);
  //     cv::imwrite(cv::format("inputs/%s.png", image_name.c_str()), IMAGES_LIST[n]);
  //     // cout << "Writing " << format("outputs/sr_%d_%d.png", row_index, col_index) << endl;
  //     // cout << "Writing " << cv::format("inputs/%s.png", image_name.c_str()) << endl;
  //     // Display the number of image processed in live
  //     //cout << "[SR7 INFO] Processed image " << n+1 << " out of " << num_of_images << endl;
  //     }
  //   cout << "\n" << endl;
  //   }
  

  total_time += (double) prepr_time.count();
  total_time += (double) dpu_time.count();
  //cout << "[TOTAL Computation Time (DPU+CPU)        ] " << total_time  << "us" << endl;
  cout << "[Average FPS with pre- & post-processing ] " << num_of_images*1000000.0/total_time  << "us" << endl;


  /////////////////////////////////////////////////////////////////////////////////////////////

  // cout << "deleting softmax     memory" << endl;
  // delete[] softmax;
  cout << "deleting inputBuffer memory" << endl;
  delete[] inputBuffer;
  cout << "deleting outputBuffer    memory" << endl;
  delete[] outputBuffer;
}
