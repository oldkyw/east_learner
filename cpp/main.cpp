
#include <iostream>
#include <vector>
#include <tuple>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


std::vector<int> get_shape(torch::Tensor tensor) {
    std::vector<int> result;
    for (int i = 0; i < tensor.ndimension(); i++)
        result.push_back(tensor.size(i));
    return result;
}


/**
 * Converts tensor to cv::Mat. 
 * The tensor size must have between 1 and 4 dimensions.
 *
 * @tensor input tensor
 * @channel_first if the input tensor is in CHW notation
 * @return cv::Mat
 */
cv::Mat tensor_to_mat(torch::Tensor tensor, bool channel_first=true) {

    int ndims = tensor.ndimension();
    int dim_offset = 0;
    if (ndims<=0 or ndims>4) {
        throw std::invalid_argument("To convert tensor to cv::Mat it has to have between 1 and 4 dims\n");
    }
    
    int n = 0, c = 0, h = 0, w = 0;
    if (channel_first and ndims>=3) {
        tensor = tensor.transpose(-3, -1);
    } 
    if (ndims==2) {
        tensor = tensor.unsqueeze(-1);
    }
    if (ndims == 4) {
        n = tensor.size(-4);
    }
    h = tensor.size(-3);
    w = tensor.size(-2);
    c = tensor.size(-1);
    
    
    if (c * h * w == 0) {
        throw std::invalid_argument("c, h, w values can't be zero\n");
    }

    int type_id = -1;
    if (tensor.dtype() == torch::kUInt8)   /*CV_8U*/  type_id = 0 + (c - 1) * 8;
    if (tensor.dtype() == torch::kInt8)    /*CV_8S*/  type_id = 1 + (c - 1) * 8;
    // if (tensor.dtype() == torch::kUInt16)    /*CV_16U*/  type_id = 2 + (c - 1) * 8;
    if (tensor.dtype() == torch::kInt16)   /*CV_16S*/ type_id = 3 + (c - 1) * 8;
    if (tensor.dtype() == torch::kInt32)   /*CV_32S*/ type_id = 4 + (c - 1) * 8;
    if (tensor.dtype() == torch::kFloat32) /*CV_32F*/ type_id = 5 + (c - 1) * 8;
    if (tensor.dtype() == torch::kFloat64) /*CV_64F*/ type_id = 6 + (c - 1) * 8;
    if (type_id == -1) {
        throw std::invalid_argument("tensor data type not supported\n");
    }

    auto out_mat = cv::Mat(cv::Size(h, w), type_id);
    auto elemsize = out_mat.elemSize1();
    std::memcpy(out_mat.data, tensor.transpose(-3,-2).contiguous().data_ptr(),  elemsize*tensor.numel());
    return out_mat;
}


/**
 * Converts cv::Mat type image data to torch tensor. 
 * Automatically checks for sizes and number of channels.
 * Optionally can flip the output tensor using channel_first to switch from HWC to CHW (fastai default) notation.
 * 
 * @cv_mat input cv matrix
 * @channel_first if the output tensor should be flipped to CHW notation
 * @squeeze should the output tensor be squeezed if number of channels C == 1
 * @return torch::tensor
 */
torch::Tensor mat_to_tensor(cv::Mat cv_mat, bool channel_first=true, bool squeeze=true) {

    int h = cv_mat.size[0];
    int w = cv_mat.size[1];
    int c = cv_mat.channels();
    auto elemsize = cv_mat.elemSize1();

    torch::TensorOptions options;
    int type_id = cv_mat.depth() % 8;
    if (type_id < 0  or type_id == 2 or type_id > 6) {
        std::cout << "Unknown mat type. Cannot convert to tensor\n";
    }
    switch (type_id) {
        // Commented out dtypes don't currently have equivalents in torch
        case 0 /*CV_8U*/: options = options.dtype(torch::kUInt8); break;
        case 1 /*CV_8S*/: options = options.dtype(torch::kInt8); break;
        // case 2 /*CV_16U*/: options = options.dtype(torch::kUInt16); break;
        case 3 /*CV_16S*/: options = options.dtype(torch::kInt16); break;
        case 4 /*CV_32S*/: options = options.dtype(torch::kInt32); break;
        case 5 /*CV_32F*/: options = options.dtype(torch::kFloat32); break;
        case 6 /*CV_64F*/: options = options.dtype(torch::kFloat64); break;
    }
    torch::Tensor result = torch::empty({h, w, c}, options);
    std::memcpy(result.data_ptr(), cv_mat.data, elemsize*result.numel());
    result = result.transpose(-3,-2);
    if (channel_first) {
        result = result.transpose(-3, -1);
    }
    if (squeeze) {
        result = result.squeeze();
    }
    return result;
}


/**
 * prepare image loaded by opencv and converted to tensor.
 * Tensor data is converted to float32 and normalized 0..1.
 * @input input tensor
 * @return tensor ready for model forward
 */
torch::Tensor prepare_image_data(torch::Tensor input) {
    auto tensor = input.to(torch::kFloat32).flip(0).unsqueeze(0);
    return tensor.div(255.0);
}


/**
 * save output as image.
 * the image gets rotated by transposition of h, w.
 * @tensor source tensor
 * @layer layer indice to store
 * @save_path output image path
 */
void save_output_layer(torch::Tensor tensor, int layer, std::string save_path) {
    auto slice = tensor.slice(1, layer, layer+1).squeeze().mul(255.0).transpose(-1,-2).to(torch::kUInt8).contiguous();
    auto slice_cv = tensor_to_mat(slice);
    cv::applyColorMap(slice_cv, slice_cv, cv::COLORMAP_PARULA);
    cv::imwrite(save_path, slice_cv);
}


class Polygon 
{
    private:
        torch::Tensor flow;
        torch::Tensor map;
        std::vector<int> size;
        float score;
        int votes;
        bool transformed;
    
    public: 
        Polygon(torch::Tensor flow, std::vector<int> size, float score=1.0, int votes=1) {
            this->flow = flow;
            this->size = size;
            this->score = score;
            this->votes = votes;
            this->transformed = false;
            this->map = this->calculate_map();
            std::cout << "Polygon was created\n";
        }

        void merge(Polygon other) {
            this->transformed = true;
            assert(this->size == other.size);
            float score_ratio = other.score / this->score;
            this->score += other.score;
            this->flow.mul_(1 / (1 + score_ratio)).add_(other.flow.mul(1 / (1 + 1 / score_ratio)));
            this->votes += other.votes;
        }

        torch::Tensor calculate_map() {
            std::cout << "Calculating map\n";
            auto points = this->flow;
            std::vector<int> hw;
            int scale_factor = 1;

            //Check if reshaping is valid (no rounding needed)
            assert(this->size[0]%scale_factor==0);
            assert(this->size[1]%scale_factor==0); 

            //Expecting only one polygon in an image pixel
            // std::cout << points.sizes() << "\n";
            assert(points.size(0)==8);


            hw.push_back(size[0]/scale_factor);
            hw.push_back(size[1]/scale_factor);
            points.div_(scale_factor);
            points = points.round().cpu().view({4, 2}).to(torch::kInt32);
            auto points_mat = tensor_to_mat(points.transpose(0,1));

            /* TODO:
            - check if the CV type is correct for mat to tensor conversion
            - change the color fill value from 1..255
            */
            cv::Mat map = cv::Mat::zeros(hw[0], hw[1], CV_8S);
            cv::fillConvexPoly(map, points_mat, 1);
            return mat_to_tensor(map);
        }

        torch::Tensor get_map() {
            if (this->transformed) {
                this->map = this->calculate_map();
            }
            return this->map;
        }

        int get_votes() const {
            return this->votes;
        }

        torch::Tensor get_flow() const {
            return this->flow;
        }
};


float iou_single_vs_single(Polygon a, Polygon b) {
    auto inter_ = (a.get_map() * b.get_map()).nonzero().numel();
    auto union_ = (a.get_map() + b.get_map()).nonzero().numel();
    if (union_==0) {
        return 0.0;
    } else {
        return inter_/union_;
    }
}

torch::Tensor iou_single_vs_many(Polygon &a, std::vector<Polygon> b) {
    if (b.size()==0) return torch::empty(0);
    std::vector<torch::Tensor> b_maps;
    for (Polygon &polygon: b) {
        b_maps.push_back(polygon.get_map());
    }
    auto a_maps_stacked = a.get_map().repeat({int(b.size()), 1, 1});
    auto b_maps_stacked = torch::stack(b_maps);

    auto inter_ = ((a_maps_stacked * b_maps_stacked) > 0).sum({1,2}, false, torch::kFloat);
    auto union_ = ((a_maps_stacked + b_maps_stacked) > 0).sum({1,2}, false, torch::kFloat);
    return inter_/union_;
}


/** 
 * Function that applies Locally Aware NMS and merges polygons given in proposals if their IoU
 * is greater than iou_threshold.
 * @min_votes minimum number of pixels voting for given polygon
 * @return a vector of merged polygons.
 */
std::vector<Polygon> lanms(std::vector<Polygon> proposals, float iou_threshold=0.5, int min_votes=4) {
    auto left = proposals;
    bool merged = true;

    while (merged) {
        merged = false;
        std::vector<Polygon> right;
        
        for (Polygon &this_: left) {
            auto res = iou_single_vs_many(this_, right);
            res *= (res >= iou_threshold).to(torch::kFloat);
            // check if there is at least one polygon meeting the criteria
            if (res.nonzero().numel() > 0) {
                int match = *std::get<1>(res.max(0)).data_ptr<long>();
                right[match].merge(this_);
                merged = true;
            } else {
                right.push_back(this_);
            }
        }
        // replace working list with the one already merged
        left = right;
    }
    
    //Filter polygons by the number of votes
    std::vector<Polygon> output;
    for (auto &polygon: left) {
        if (polygon.get_votes() >= min_votes) {
            output.push_back(polygon);
        }
    }
    return output;
}


/**
 * Converts prediction tensor (model output) to vector of Polygons
 * @min_score score threshold (max=1.0)
 * @return vector of Polygons
 */
std::vector<Polygon> pred2poly(torch::Tensor pred, float min_score=0.8) {
    if (pred.ndimension()==4) {
        if (pred.size(0)==1) {
            pred.squeeze_();
        } else {
            throw std::invalid_argument("4 dimensional prediction can only be a single element batch\n");
        }
    }

    int c, h, w;
    c = pred.size(0);
    h = pred.size(1);
    w = pred.size(2);

    // t = pred.clone()
    auto options = torch::TensorOptions().device(torch::kCPU);
    auto lr = torch::linspace(0, w - 1, w, options);
    auto tb = torch::linspace(0, h - 1, h, options).view({-1, 1});
    // skip score mask & select just corners geometry from the prediction tensor 1:8
    auto geom = pred.narrow(-3, 1, c - 1);
    
    // run operations on geometry to recover location of corners
    // add decremental linspace left right for x_dist channels
    // geom[::2]
    geom.slice(0, 0, geom.size(0), 2).mul_(-w).add_(lr);
    // add decremental linspace top bottom for y_dist channels
    // geom[1::2]
    geom.slice(0, 1, geom.size(0), 2).mul_(-h).add_(tb);

    // transpose and reshape from B, C, H, W to B, H*W, C
    pred = pred.transpose(-2, -1).transpose(-3, -1);
    pred = pred.view({h * w, c});
    // t = t.view(-1, size * size, n_c); filtered = [ti[ti[:, 0] > min_score] for ti in t]

    auto indices = (pred.slice(1, 0, 1, 1) > min_score).nonzero();
    pred = pred.index_select(0, indices.select(1,0));

    std::vector<int> size = {h, w};
    std::vector<Polygon> polygons;
    torch::Tensor flow;
    float score;

    for (int i=0; i<pred.size(0); i++){
        flow = pred[i].slice(0,1,9,1);
        score = *pred[i][0].data_ptr<float>();
        polygons.push_back(Polygon(flow, size, score));
    }
    return polygons;
}


int main(int argc, const char* argv[]) {
    // if (argc != 3) {
    //     std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    
    //     return -1;
    // }
    // auto model_path = argv[1];
    // auto img_path = argv[2];
    auto model_path = "./models/pvanet_traced.pt";
    auto img_path = "./dataset/test/img/82.jpg";
    torch::jit::script::Module module;
    std::vector<torch::jit::IValue> input;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "Model loaded successfully.\n";

    cv::Mat img = cv::imread(img_path);
    cv::resize(img, img, cv::Size(512,512));
    
    auto tensor = prepare_image_data(mat_to_tensor(img));
    input.push_back(tensor);
    std::cout << "prepared tensor: " << tensor << std::endl;

    torch::Tensor output = module.forward(input).toTensor();
    // auto preds = ;
    auto detections = lanms(pred2poly(output));
    for(const Polygon &polygon: detections) {
        std::cout << "votes: " << polygon.get_votes() << " flow:" << polygon.get_flow() << "\n"; 
    }
    save_output_layer(output, 0, "./score.jpg");
}
