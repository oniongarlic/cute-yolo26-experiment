#include "yolo26models.h"

Yolo26Models::Yolo26Models()
{
#ifdef WIN32
    basepath="C:/Development/projects/qtopencv/models/";
#else
    basepath="/data/repos/models/";
#endif

    models = {
        {"yolo26n.onnx","Yolo26n"},
        {"yolo26n-pose.onnx","Yolo26n Pose"},
        {"yolo26n-seg.onnx","Yolo26n Seg"},

        {"yolo26s.onnx","Yolo26s"},
        {"yolo26s-pose.onnx","Yolo26s pose"},
        {"yolo26s-seg.onnx","Yolo26s seg"},

        {"yolo26m.onnx","Yolo26m"},
        {"yolo26m-pose.onnx","Yolo26m pose"},
        {"yolo26m-seg.onnx","Yolo26m seg"},

        {"yolo26l.onnx","Yolo26l"},
        {"yolo26l-pose.onnx","Yolo26l pose"},
        {"yolo26l-seg.onnx","Yolo26l seg"},

        {"yolo26n-depth.onnx","Yolo26 Depth"}
    };
}

cv::dnn::Net &Yolo26Models::net(int idx)
{
    return nets.at(idx);
}

std::string &Yolo26Models::name(int idx)
{
    return models.at(idx).name;
}

void Yolo26Models::setBasepath(const std::string path)
{
    basepath=path;
}

void Yolo26Models::load()
{
    Model m;

    foreach (m, models) {
        qDebug() << "Loading " << m.onnx << " from " << basepath;
        try {
            auto net=cv::dnn::readNetFromONNX(basepath+m.onnx);
            nets.push_back(net);
        } catch (const cv::Exception& ex) {
            qWarning() << "Failed to load model " << m.name << ex.codeMessage() << ex.what();
        }
    }
}

