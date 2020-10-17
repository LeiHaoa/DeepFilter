#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>


int main()
{
    // 加载模型
    //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("checkpoint_fastvc_20-10-06-20-26-58_ecpch10.pt");
    torch::jit::script::Module module = torch::jit::load("/home/haoz/deepfilter/cpp/checkpoint_fastvc_20-10-06-20-26-58_ecpch10.pt");
    // 转化为CPU模型
    module.to(at::kCPU);
    std::cout << "========================= Load model is ok!! ====================\n" << std::endl;
	

	// create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({2,21}));
	

    // forward
    auto output = module.forward(inputs).toTensor();

    for (int i = 0; i < 2; ++i)
    {
        // 转化成Float
		for(int j = 0; j < 2; ++j){
				std::cout << output[i][j].item().toFloat() <<",";
		}
		std::cout << std::endl;
	}
    
    return 0;
}


