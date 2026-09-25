#ifndef YOLO26MODELS_H
#define YOLO26MODELS_H

#include <opencv2/dnn.hpp>

#include <QDebug>

struct Model {
    std::string onnx;
    std::string name;
};

class Yolo26Models
{
public:
    Yolo26Models(std::vector<Model> models);
    void load();

    cv::dnn::Net &net(int idx);
    std::string &name(int idx);

    void setBasepath(const std::string path);
    void setGpu(bool gpu) {m_gpu=gpu;}

    const std::string basepath() { return m_basepath; }

private:
    std::string m_basepath;
    std::vector<Model> m_models;
    std::vector<cv::dnn::Net> m_nets;
    bool m_gpu=false;
};

#endif // YOLO26MODELS_H
