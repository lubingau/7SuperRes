#ifndef DEBUG_H
#define DEBUG_H

#define DEBUG_EVAL_PSNR 0 // Print PSNR for each image  
#define DEBUG_EVAL_SAVE 0 // Save input and output of eval into debug/eval/input and debug/eval/output

#define DEBUG_PATCHER 0 // Save patches into debug/patcher folder
#define DEBUG_RUNCNN 0 || DEBUG_EVAL_SAVE // Save input and output of runCNN into debug/runCNN/input and debug/runCNN/output
#define DEBUG_REBUILDER 0 // Save sum_image and mask into debug/rebuilder folder

#endif
