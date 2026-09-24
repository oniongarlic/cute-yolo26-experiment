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
    Yolo26Models();
    void load();

    cv::dnn::Net &net(int idx);
    std::string &name(int idx);

    void setBasepath(const std::string path);

private:
    std::string basepath;
    std::vector<Model> models;
    std::vector<cv::dnn::Net> nets;
};

#endif // YOLO26MODELS_H
