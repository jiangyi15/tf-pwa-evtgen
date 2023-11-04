// CppFlow headers
// #include <cppflow/model.h>
// #include <cppflow/ops.h>

#include <tensorflow/c/c_api.h>
// C++ headers
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

inline bool status_check(TF_Status *status) {
  if (TF_GetCode(status) != TF_OK) {
    throw std::runtime_error(TF_Message(status));
  }
  return true;
}

class Tensor {
public:
  Tensor(std::vector<double> p) {
    auto data = p.data();
    std::vector<int64_t> shape = {(int64_t)p.size()};
    auto len = p.size() * sizeof(double);
    this->tf_tensor = {TF_AllocateTensor(TF_DOUBLE, shape.data(),
                                         static_cast<int>(shape.size()), len),
                       TF_DeleteTensor};
    memcpy(TF_TensorData(this->tf_tensor.get()), data,
           TF_TensorByteSize(this->tf_tensor.get()));
  }

  std::shared_ptr<TF_Tensor> get_tensor() const {
    return tf_tensor;
  }
  Tensor(TF_Tensor *t) {
    this->tf_tensor = {t, TF_DeleteTensor};
  }

  std::vector<double> get_data() {
    auto res_tensor = get_tensor();

    // Check tensor data is not empty
    auto raw_data = TF_TensorData(res_tensor.get());
    // this->error_check(raw_data != nullptr, "Tensor data is empty");

    size_t size = (TF_TensorByteSize(res_tensor.get()) /
                   TF_DataTypeSize(TF_TensorType(res_tensor.get())));

    // Convert to correct type
    const auto T_data = static_cast<double *>(raw_data);
    std::vector<double> r(T_data, T_data + size);

    return r;
  }

  mutable std::shared_ptr<TF_Tensor> tf_tensor;
};

std::string parse_name_string(const std::string &name) {
  auto idx = name.find(':');
  return (idx == std::string::npos ? name : name.substr(0, idx));
}

int parse_name_index(const std::string &name) {
  auto idx = name.find(':');
  return (idx == std::string::npos ? 0 : std::stoi(name.substr(idx + 1)));
}

class Model {
public:
  Model(std::string filename) {

    this->graph = TF_NewGraph();
    std::shared_ptr<TF_Status> status = {TF_NewStatus(), &TF_DeleteStatus};
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>
        session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {
        TF_NewBufferFromString("", 0), TF_DeleteBuffer};
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {
        TF_NewBuffer(), TF_DeleteBuffer};
    int tag_len = 1;
    const char *tag = "serve";
    this->session = TF_LoadSessionFromSavedModel(
        session_options.get(), run_options.get(), filename.c_str(), &tag,
        tag_len, this->graph, meta_graph.get(), status.get());
    status_check(status.get());
  }

  ~Model() {
    std::shared_ptr<TF_Status> status = {TF_NewStatus(), &TF_DeleteStatus};
    TF_DeleteGraph(this->graph);
    TF_DeleteSession(this->session, status.get());
    status_check(status.get());
  }

  std::vector<Tensor>
  operator()(std::vector<std::tuple<std::string, Tensor>> inputs,
             std::vector<std::string> outputs) {

    std::vector<TF_Output> inp_ops(inputs.size());
    std::vector<TF_Tensor *> inp_val(inputs.size(), nullptr);

    for (decltype(inputs.size()) i = 0; i < inputs.size(); i++) {
      // Operations
      const auto op_name = parse_name_string(std::get<0>(inputs[i]));
      const auto op_idx = parse_name_index(std::get<0>(inputs[i]));
      inp_ops[i].oper = TF_GraphOperationByName(this->graph, op_name.c_str());
      inp_ops[i].index = op_idx;

      if (!inp_ops[i].oper)
        throw std::runtime_error("No operation named \"" + op_name +
                                 "\" exists");

      // Values
      inp_val[i] = std::get<1>(inputs[i]).get_tensor().get();
    }

    std::vector<TF_Output> out_ops(outputs.size());
    auto out_val = std::vector<TF_Tensor *>(outputs.size());
    for (decltype(outputs.size()) i = 0; i < outputs.size(); i++) {
      const auto op_name = parse_name_string(outputs[i]);
      const auto op_idx = parse_name_index(outputs[i]);
      out_ops[i].oper = TF_GraphOperationByName(this->graph, op_name.c_str());
      out_ops[i].index = op_idx;

      if (!out_ops[i].oper)
        throw std::runtime_error("No operation named \"" + op_name +
                                 "\" exists");
    }

    std::shared_ptr<TF_Status> status = {TF_NewStatus(), &TF_DeleteStatus};
    TF_SessionRun(
        this->session, /*run_options*/ NULL, inp_ops.data(), inp_val.data(),
        static_cast<int>(inputs.size()), out_ops.data(), out_val.data(),
        static_cast<int>(outputs.size()),
        /*targets*/ NULL, /*ntargets*/ 0, /*run_metadata*/ NULL, status.get());
    status_check(status.get());

    std::vector<Tensor> result;
    result.reserve(outputs.size());
    for (decltype(outputs.size()) i = 0; i < outputs.size(); i++) {
      result.emplace_back(Tensor(out_val[i]));
    }

    return result;
  }

  TF_Graph *graph;
  TF_Session *session;
};

int main() {
  Model model("model2/");
  
  for (int i = 0; i < 10; i++) {

    auto p_0 = Tensor({2.19771882, 0.73008701, -0.55660393, -0.70110049});
    auto p_1 = Tensor({0.77582355, -0.19216696, 0.4872549, -0.2895509});
    auto p_2 = Tensor({2.12028122, -0.48871821, 0.04448316, 0.8815766});
    auto p_3 = Tensor({0.18551641, -0.04920183, 0.02486587, 0.1090748});
    
    std::vector<std::tuple<std::string, Tensor>> inputs;
    inputs.push_back(std::tuple<std::string, Tensor>("serving_default_p_0:0",p_0));
    inputs.push_back(std::tuple<std::string, Tensor>("serving_default_p_1:0",p_1));
    inputs.push_back(std::tuple<std::string, Tensor>("serving_default_p_2:0",p_2));
    inputs.push_back(std::tuple<std::string, Tensor>("serving_default_p_3:0",p_3));


    
    auto output = model(inputs,
                        {
                            "StatefulPartitionedCall:0",
                        });

    auto data = output[0].get_data();

    std::cout << "weight: " << data[0] << std::endl;
  }
  return 0;
}
