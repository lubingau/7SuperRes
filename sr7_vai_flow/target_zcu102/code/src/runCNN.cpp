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
#include <opencv2/opencv.hpp>

#include "debug.h"

using namespace std;
using namespace cv;
using namespace std::chrono;


void extrapolateImages(const vector<Mat>& inputImages, vector<Mat>& outputImages) {
    cout << "[SR7 INFO RunCNN] Interpolating images..." << endl;
    for (unsigned int n=0; n<inputImages.size(); n++) {
        Mat outputImage;
        Mat inputImage = inputImages[n];
        resize(inputImage, outputImage, Size(), 2.0, 2.0, INTER_LINEAR);
        outputImages.push_back(outputImage);
    }
    cout << "[SR7 INFO RunCNN] Interpolation done!" << endl;
}

void runDPU(vart::Runner *runner, int8_t *inputBuffer, int8_t *outputBuffer, GraphInfo *shapes, int num_images_x_thread)
{
  // get in/out tensors and dims
  auto outputTensors = runner->get_output_tensors();
  // size of the output tensor
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();
  // get shape info
  int outSize = shapes->outTensorList[0].size;
  int outHeight = shapes->outTensorList[0].height;
  int outWidth = shapes->outTensorList[0].width;
  int inSize = shapes->inTensorList[0].size;
  int inHeight = shapes->inTensorList[0].height;
  int inWidth = shapes->inTensorList[0].width;
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  int8_t *loc_inputBuffer = inputBuffer;
  int8_t *loc_outputBuffer = outputBuffer;

  cout << "[SR7 INFO AI] Inside RUN CNN " << endl;

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


void runCNN(const vector<Mat>& inputImages, vector<Mat>& outputImages, const string xmodel_path, int num_threads
#if DEBUG_RUNCNN
    , const string& input_folder, const string& output_folder
#endif
) {
  assert((num_threads <= 6) & (num_threads >= 1));
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPARE DPU STUFF

  auto graph = xir::Graph::deserialize(xmodel_path);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "[SR7 INFO AI] Create running for subgraph: " << subgraph[0]->get_name();
  cout << "[SR7 INFO AI] Create running for subgraph: " << subgraph[0]->get_name() << endl;
  
  int num_of_images = 0;
  int num_images_x_thread = 0;
  int num_images_first_thread = 0;

  GraphInfo shapes;
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
  cout << "---------------------------------------------" << endl;
  cout << "[SR7 INFO AI] outSize  " << outSize << endl;
  cout << "[SR7 INFO AI] inSize   " << inSize << endl;
  cout << "[SR7 INFO AI] outW     " << outWidth << endl;
  cout << "[SR7 INFO AI] outH     " << outHeight << endl;
  cout << "[SR7 INFO AI] inpW     " << inWidth << endl;
  cout << "[SR7 INFO AI] inpH     " << inHeight << endl;
  cout << "[SR7 INFO AI] class/channel  " << num_of_classes << endl;
  cout << "[SR7 INFO AI] batchSize      " << batchSize << endl; // alway 1 for Edge
  cout << "[SR7 INFO AI] in_fixpos      " << in_fixpos << endl;
  cout << "[SR7 INFO AI] in_fix_scale   " << in_fix_scale << endl;
  cout << "[SR7 INFO AI] input_scale   " << input_scale << endl;
  cout << "[SR7 INFO AI] out fix scale  " << out_fix_scale << endl;
  cout << "[SR7 INFO AI] output_scale   " << output_scale << endl;
  cout << "---------------------------------------------" << endl;


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
  cout << "[SR7 INFO AI] Average calibration high resolution clock: " << avg_calibr_highres.count() / num_of_trials << "us"  << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY ALLOCATION

  num_of_images = inputImages.size();
  cout << "[SR7 INFO AI] Number of images to read " << num_of_images << endl;

  // number of images per thread
  num_images_x_thread = num_of_images / num_threads;
  num_images_first_thread = num_of_images - (num_images_x_thread * (num_threads-1));
  cout << "[SR7 INFO AI] Number of images in the first thread: " << num_images_first_thread << endl;
  cout << "[SR7 INFO AI] Number of images per thread: " << num_images_x_thread << endl;

  // memory allocation
  Mat image = cv::Mat(inHeight, inWidth, CV_8UC3);
  Mat debug = cv::Mat(outHeight, outWidth, CV_8UC3);
  
  int8_t *inputBuffer = new int8_t[(num_of_images)*inSize];
  int8_t *outputBuffer    = new int8_t[(num_of_images)*outSize];

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPROCESSING ALL IMAGES AT ONCE
  cout << "[SR7 INFO AI] Start pre-processing" << endl;
  auto pre_t1 = std::chrono::high_resolution_clock::now();
  
  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = inputImages[n];

      for (int y = 0; y < inHeight; y++) {
	       for (int x = 0; x < inWidth; x++) {
            for (int c = 0; c < 3; c++) {
              float tmp_pix = ((float) image.at<Vec3b>(y,x)[c])/255;
              tmp_pix = tmp_pix * input_scale;
              inputBuffer[n*inSize + 3*(y*inWidth+x) + c  ] = (int8_t) tmp_pix; //BGR format
              //inputBuffer[n*inSize + 3*(y*inWidth+x) + 2-c] = (int8_t) tmp_pix; //RGB format
            }
	       }
      }
#if DEBUG_RUNCNN
      // Only for debug
      Mat debug = cv::Mat(inHeight, inWidth, CV_8UC3);
      for (int i = 0; i < inHeight; i++){
        for (int j = 0; j < inWidth; j++){
          int B_pix = inputBuffer[n*inSize + 3*(i*inWidth+j) + 0];
          int G_pix = inputBuffer[n*inSize + 3*(i*inWidth+j) + 1];
          int R_pix = inputBuffer[n*inSize + 3*(i*inWidth+j) + 2];
          debug.at<Vec3b>(i, j) = Vec3b(B_pix, G_pix, R_pix);
        }
      }
      string filename = input_folder + "debug_in_" + to_string(n) + ".png";
      imwrite(filename, debug);
      cout << "[SR7 INFO] Image " << n << " pre-processed" << endl;
#endif
  }
  cout << "[SR7 INFO AI] Images loaded in the buffer" << endl;

  auto pre_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_time = pre_t2 - pre_t1 - avg_calibr_highres;
  cout << "---------------------------------------------" << endl;
  cout << "[SR7 INFO AI] Preprocess time: " << 1000.0*prepr_time.count() << "ms" << endl;
  cout << "[SR7 INFO AI] Preprocess FPS:  " << num_of_images*1000000.0/prepr_time.count()  << endl;
  cout << "---------------------------------------------" << endl;
  

  // split images in chunks, each chunks for its own thead
  // avoid pointing to wrong memorycv::Mat> locations
  int8_t *inputBuffer0, *inputBuffer1, *inputBuffer2, *inputBuffer3, *inputBuffer4, *inputBuffer5;
  int8_t *outputBuffer0, *outputBuffer1, *outputBuffer2, *outputBuffer3, *outputBuffer4, *outputBuffer5;

  if (num_threads >= 1) {
      inputBuffer0 = inputBuffer + inSize * 0 * num_images_first_thread;
      outputBuffer0 = outputBuffer + outSize * 0 * num_images_first_thread;
  }
  if (num_threads >= 2) {
      inputBuffer1 = inputBuffer + inSize * 1 * num_images_x_thread;
      outputBuffer1 = outputBuffer + outSize * 1 * num_images_x_thread;
  }
  if (num_threads >= 3) {
      inputBuffer2 = inputBuffer + inSize * 2 * num_images_x_thread;
      outputBuffer2 = outputBuffer + outSize * 2 * num_images_x_thread;
  }
  if (num_threads >= 4) {
      inputBuffer3 = inputBuffer + inSize * 3 * num_images_x_thread;
      outputBuffer3 = outputBuffer + outSize * 3 * num_images_x_thread;
  }
  if (num_threads >= 5) {
      inputBuffer4 = inputBuffer + inSize * 4 * num_images_x_thread;
      outputBuffer4 = outputBuffer + outSize * 4 * num_images_x_thread;
  }
  if (num_threads >= 6) {
      inputBuffer5 = inputBuffer + inSize * 5 * num_images_x_thread;
      outputBuffer5 = outputBuffer + outSize * 5 * num_images_x_thread;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MULTITHREADING DPU EXECUTION WITH BATCH
  cout << "[SR7 INFO AI] Start DPU execution" << endl;
  thread workers[num_threads];

  auto dpu_t1 = std::chrono::high_resolution_clock::now();

  for (auto i = 0; i < num_threads; i++) {
    if (i == 0)
      workers[i] = thread(runDPU, runner.get(), ref(inputBuffer0), ref(outputBuffer0), &shapes, num_images_first_thread);
    if (i == 1)
      workers[i] = thread(runDPU, runner1.get(), ref(inputBuffer1), ref(outputBuffer1), &shapes, num_images_x_thread);
    if (i == 2)
      workers[i] = thread(runDPU, runner2.get(), ref(inputBuffer2), ref(outputBuffer2), &shapes, num_images_x_thread);
    if (i == 3)
      workers[i] = thread(runDPU, runner3.get(), ref(inputBuffer3), ref(outputBuffer3), &shapes, num_images_x_thread);
    if (i == 4)
      workers[i] = thread(runDPU, runner4.get(), ref(inputBuffer4), ref(outputBuffer4), &shapes, num_images_x_thread);
    if (i == 5)
      workers[i] = thread(runDPU, runner5.get(), ref(inputBuffer5), ref(outputBuffer5), &shapes, num_images_x_thread);
  }
  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }
  cout << "[SR7 INFO AI] DPU execution done!" << endl;

  auto dpu_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> dpu_time = dpu_t2 - dpu_t1 - avg_calibr_highres;
  cout << "---------------------------------------------" << endl;
  cout << "[SR7 INFO AI] DPU total time: " << 1000.0*dpu_time.count() << "ms" << endl;
  //cout << "[DPU avg Time ] " << (dpu_time.count()/num_of_images) << "us" << endl;
  cout << "[SR7 INFO AI] DPU avg FPS:    " << num_of_images*1000000.0/dpu_time.count() << endl;
  cout << "---------------------------------------------" << endl;

  double total_time = 0.0;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // READ OUTPUT BUFFER
  cout << "[SR7 INFO AI] Start post-processing" << endl;
  cout << "[SR7 INFO AI] Read output buffer" << endl;

  vector<Mat> outputImages_temp;

  // Create a Mat to hold the super-resolved image
  Mat superResolvedImage(outHeight, outWidth, CV_8UC3);
  int B_pix, G_pix, R_pix;
  for (unsigned int n = 0; n < num_of_images; n++) {
    // Iterate over rows and columns of the super-resolved image
    for (int row = 0; row < outHeight; row++) {
      for (int col = 0; col < outWidth; col++) {
        B_pix = 2*outputBuffer[n*outSize + 3*(row*outWidth+col) + 0];
        G_pix = 2*outputBuffer[n*outSize + 3*(row*outWidth+col) + 1];
        R_pix = 2*outputBuffer[n*outSize + 3*(row*outWidth+col) + 2];
        superResolvedImage.at<cv::Vec3b>(row, col) = Vec3b(B_pix, G_pix, R_pix);
      }
    }
    outputImages.push_back(superResolvedImage.clone());
  }
  cout << "[SR7 INFO AI] Post-processing done!" << endl; 

  total_time += (double) prepr_time.count();
  total_time += (double) dpu_time.count();
  cout << "[SR7 INFO AI] TOTAL Computation Time (DPU+CPU):   " << 1000.0*total_time  << "ms" << endl;
  cout << "[SR7 INFO AI] Average FPS of AI:                  " << num_of_images*1000000.0/total_time << endl;

#if DEBUG_RUNCNN
  // Only for debug
  for (unsigned int n = 0; n < num_of_images; n++) {
    debug = outputImages[n];
    for (int i = 0; i < outHeight; i++){
      for (int j = 0; j < outWidth; j++){
        B_pix = debug.at<Vec3b>(i, j)[0];
        G_pix = debug.at<Vec3b>(i, j)[1];
        R_pix = debug.at<Vec3b>(i, j)[2];
        debug.at<Vec3b>(i, j) = Vec3b(B_pix, G_pix, R_pix);
      }
    }
    string filename = output_folder + "debug_out_" + to_string(n) + ".png";
    imwrite(filename, debug);
    cout << "[SR7 INFO AI] Image " << n << " super-resolved" << endl;
  }
#endif

  /////////////////////////////////////////////////////////////////////////////////////////////
  // FREE MEMORY
  cout << "[SR7 INFO AI] Deleting inputBuffer memory" << endl;
  delete[] inputBuffer;
  cout << "[SR7 INFO AI] Deleting outputBuffer memory" << endl;
  delete[] outputBuffer;
}
