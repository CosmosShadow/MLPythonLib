// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "ios_image_load.h"

NSString* RunInferenceOnImage();

namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
    NSString* inference_result = RunInferenceOnImage();
    self.urlContentTextView.text = inference_result;
}

@end

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                             new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

NSString* RunInferenceOnImage() {
    // 创建graph
    tensorflow::GraphDef tensorflow_graph;
    NSString* network_path = FilePathForResourceName(@"mnist.quantized", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    // 根据读取的模型创建session
    tensorflow::SessionOptions options;
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return [NSString stringWithFormat: @"Session create failed - %s",
                status_string.c_str()];
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return @"";
    }
    
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({1, 784}));
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{"x", image_tensor}}, {"output"}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return @"Error";
    }
        
    tensorflow::Tensor* output = &outputs[0];
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = output->flat<float>();
    const int count = (int)prediction.size();
    // 5种可能的猜测
    for (int i = 0; i < count; ++i) {
        const float value = prediction(i);
        LOG(INFO) << value << "\n";
    }
    
    
    NSString* result = @"shit";
    return result;
}
