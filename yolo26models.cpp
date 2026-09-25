#include "yolo26models.h"

Yolo26Models::Yolo26Models(std::vector<Model> models)
{
#ifdef WIN32
    m_basepath="C:/Development/projects/qtopencv/models/";
#else
    m_basepath="/data/repos/models/";
#endif

    m_models=models;
}

cv::dnn::Net &Yolo26Models::net(int idx)
{
    return m_nets.at(idx);
}

std::string &Yolo26Models::name(int idx)
{
    return m_models.at(idx).name;
}

void Yolo26Models::setBasepath(const std::string path)
{
    m_basepath=path;
}

void Yolo26Models::load()
{
    Model m;

    foreach (m, m_models) {
        qDebug() << "Loading " << m.onnx << " from " << m_basepath;
        try {
            auto net=cv::dnn::readNetFromONNX(m_basepath+m.onnx, m_gpu ? cv::dnn::ENGINE_CLASSIC : cv::dnn::ENGINE_AUTO);
            if (m_gpu) {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
            m_nets.push_back(net);
        } catch (const cv::Exception& ex) {
            qWarning() << "Failed to load model " << m.name << ex.codeMessage() << ex.what();
        }
    }
}

