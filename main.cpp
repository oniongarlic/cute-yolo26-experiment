#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "yolo26models.h"

constexpr int target = 640;

static const std::string kWinMain = "OpenCV + YOLO26";
static const std::string kWinMask = "Mask";
static const std::string kWinDepth = "Depth";

static cv::Mat lbox = cv::Mat(target, target, CV_8UC3, cv::Scalar(114, 114, 114));

struct Detection {
    cv::Rect2f box;
    float score;
    int cls;
    std::vector<cv::Point3f> pose;
    cv::Mat mask;
};

cv::TickMeter tm;
double fps=0;
int cneti=0;

Yolo26Models models;

cv::dnn::Net *cnet;
std::string cname;
cv::dnn::Net *dnet;

void blend(const cv::Mat &bg, const cv::Mat &fg, const cv::Mat &mask, cv::Mat &res)
{
    static cv::Mat mf,m3,f1,f2;

    try {
        bg.convertTo(f1, CV_32FC1, 1.0 / 255.0);
        fg.convertTo(f2, CV_32FC1, 1.0 / 255.0);

        // blend with mask
        cv::multiply(f1, cv::Scalar(1.0, 1.0, 1.0)-m3, f1);
        cv::multiply(f2, m3, f2);

        cv::add(f1, f2, res);
    } catch (const cv::Exception& ex) {
        res=fg;
    }
}

void blendColor(cv::Mat &bg, const cv::Scalar &fg, const cv::Mat &mask)
{
    qDebug("A");
    cv::Mat roiF;
    bg.convertTo(roiF, CV_32FC3);

    qDebug("A");
    cv::Mat alpha3;
    cv::merge(std::vector<cv::Mat>{mask, mask, mask}, alpha3);
    qDebug("BBB");

    cv::Mat roiF1 = roiF.mul(1.0f - alpha3);
    qDebug("CCCCC");
    cv::Mat roiF2 = roiF.mul(alpha3);
    qDebug("B");
    //roiF=roiF1+roiF2;
    cv::Mat b=roiF1+roiF2;
    qDebug("C");
    roiF.convertTo(bg, CV_8UC3);
}

cv::Mat decodeMask(const cv::Mat& coefficients, const cv::Mat& proto, const cv::Rect2f& box640)
{
    cv::Mat mask640;

    const int maskDim     = proto.size[1];
    const int protoHeight = proto.size[2];
    const int protoWidth  = proto.size[3];

    cv::Mat protoMat = proto.reshape(1, maskDim);

    // mask coefficients × prototypes
    cv::Mat mask = coefficients * protoMat;
    mask = mask.reshape(1, protoHeight);

    // sigmoid
    cv::exp(-mask, mask);
    mask = 1.0f / (1.0f + mask);

    // Convert 640x640 detection box to prototype coordinates.
    const float sx = static_cast<float>(protoWidth) / 640.0f;
    const float sy = static_cast<float>(protoHeight) / 640.0f;

    cv::Rect2f protoBox(
        box640.x      * sx,
        box640.y      * sy,
        box640.width  * sx,
        box640.height * sy
        );

    // Crop mask from detection box
    cv::Rect crop(
        cvFloor(protoBox.x),
        cvFloor(protoBox.y),
        cvCeil(protoBox.width),
        cvCeil(protoBox.height)
        );

    crop &= cv::Rect(0, 0, protoWidth, protoHeight);

    if (crop.empty())
        return {};

    auto bmax = cv::Rect2f(0, 0, 640, 640);

    cv::resize(mask, mask640, cv::Size(640, 640));
    //cv::rectangle(mask640, box640, cv::Scalar(255, 255, 255), 2);

    return mask640(box640 & bmax);
}

cv::Mat visualize_depth(const cv::Mat& depth)
{
    cv::Mat vdepth;

    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    depth.convertTo(vdepth, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    return vdepth;
}

cv::Mat depth(cv::dnn::Net& net, const cv::Mat& frame)
{
    cv::Mat resized;

    cv::resize(frame, resized, cv::Size(768, 768));

    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, {768, 768}, cv::Scalar(), true, false);
    net.setInput(blob);

    cv::Mat dep = net.forward();

    const int height = dep.size[2];
    const int width  = dep.size[3];

    cv::Mat depthm = dep.reshape(1, height);

    cv::resize(depthm, resized, frame.size());

    return resized;
}

std::vector<Detection> detect(cv::dnn::Net& net, const cv::Mat& frame, float conf_thres, bool bin)
{
    const float scale = std::min(
        static_cast<float>(target) / frame.cols,
        static_cast<float>(target) / frame.rows
        );

    cv::Size newSize(cvRound(frame.cols * scale), cvRound(frame.rows * scale) );

    cv::Mat resized;
    cv::resize(frame, resized, newSize);

    const int dx = target - newSize.width;
    const int dy = target - newSize.height;

    const int ox = (target - resized.cols) / 2;
    const int oy = (target - resized.rows) / 2;

    resized.copyTo(lbox(cv::Rect(ox, oy, resized.cols, resized.rows)));

    cv::Mat blob = cv::dnn::blobFromImage(lbox, 1.0/255.0, {640, 640}, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);
    cv::Mat out = outputs[0];

    const int rows = out.size[out.dims - 2], cols = out.size[out.dims - 1];
    const float* p = reinterpret_cast<const float*>(out.data);

    std::vector<Detection> dets;
    for (size_t i = 0; i < rows; i++) {
        const float* row = p + (size_t)i * cols;

        if (row[4] < conf_thres)
            continue;

        Detection d;
        // Scale the 640x640 output to frame size
        d.box = cv::Rect2f(
            std::clamp((row[0]-dx/2)/scale, 0.0f, (float)frame.size().width ),
            std::clamp((row[1]-dy/2)/scale, 0.0f, (float)frame.size().height ),
            (row[2]-row[0])/scale,
            (row[3]-row[1])/scale);
        d.score = row[4];
        d.cls = (int)row[5];

        switch (cols) {
        case 6: // Detection, do nothing extra
            break;
        case 57: // Pose points (x,y,c)
            for (size_t p = 0; p < 17; p++) {
                auto pp=cv::Point3f((row[6+p*3]-dx/2)/scale, (row[6+p*3+1]-dy/2)/scale, row[6+p*3+2]);
                d.pose.push_back(pp);
            }
            break;
        case 38: // Seg mask coefficients
            if (d.cls==0)
            {
                cv::Mat mout = outputs[1];
                cv::Mat coeffs(1, 32, CV_32F);

                for (size_t p = 0; p < 32; p++) {
                    coeffs.at<float>(0, p)=row[6+p];
                }
                std::string e;

                auto b640=cv::Rect2f(
                    (row[0]),
                    (row[1]),
                    (row[2]-row[0]),
                    (row[3]-row[1]));

                // Get mask in detection box size
                d.mask=decodeMask(coeffs, mout, b640);
            }
            break;
        default:
            qDebug() << "Model with unhandled cols " << cols;
        }

        dets.push_back(d);
    }
    return dets;
}

// #define FHD 1

cv::Point2f point3to2(const cv::Point3f &p3)
{
    return cv::Point2f(p3.x, p3.y);
}

void set_current_network(int i)
{
    cnet=&models.net(i);
    tm.reset();
    fps=0;
    cneti=i;
    cname=models.name(i);
}

/**
 *
 * Draw a pose line, but only if both points are over threshold
 */
void posel(cv::Mat &frame, std::vector<cv::Point3f> pose, int f, int t, const cv::Scalar color)
{
    if (pose[f].z<0.6 || pose[t].z<0.6)
        return;

    cv::line(frame, point3to2(pose[f]), point3to2(pose[t]), color, 2);
}

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    QString file, outfile;
    int camera=0;
    cv::Mat frame;
    bool run=true, bin=false, blur=false, pred=true,contour=false, paused=false,showdepth=false;
    int f=0;
    double thres=0.6;

    QCoreApplication app(argc, argv);
    QCommandLineParser parser;

    QCommandLineOption cameraOption("c", "Camera ID", "camera");
    parser.addOption(cameraOption);

    QCommandLineOption inputOption("i", "Input video file.", "file");
    parser.addOption(inputOption);

    QCommandLineOption outputOption("o", "Output video file.", "file");
    parser.addOption(outputOption);

    QCommandLineOption modelOption("b", "Model base path.", "path");
    parser.addOption(modelOption);

    QCommandLineOption defaultOption("m", "Start with given model.", "model");
    parser.addOption(defaultOption);

    parser.process(app);

    if (parser.isSet(inputOption)) {
        file=parser.value(inputOption);
        camera=-1;
        qDebug() << "Input file " << file;
    }

    if (parser.isSet(outputOption)) {
        outfile=parser.value(outputOption);
        qDebug() << "Output file " << file;
    }

    if (parser.isSet(cameraOption)) {
        camera=parser.value(cameraOption).toInt();
        qDebug() << "Camera set to " << camera;
    }

    if (parser.isSet(modelOption)) {
        models.setBasepath(parser.value(modelOption).toStdString());
        //qDebug() << "Loading models from " << basepath;
    }

    models.load();

    set_current_network(0);
    dnet=&models.net(12);

    if (camera>-1) {
        cap.open(camera);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
#ifdef FHD
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
#else
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
#endif
    } else {
        cap.open(file.toStdString());
    }

    if (!cap.isOpened()) {
        printf("Failed to open video input\n");
        return 1;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (!outfile.isEmpty()) {
        writer.open(outfile.toStdString(), cv::VideoWriter::fourcc('X', '2', '6', '4'), 30.0, cv::Size(frame_width, frame_height));
    }

    cv::namedWindow(kWinMain, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::namedWindow(kWinMask, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::namedWindow(kWinDepth, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    while (cap.read(frame) && run) {

        tm.start();
        f++;

        auto df=detect(*cnet, frame, thres, bin);

        if (showdepth && f % 8==0) {
            auto dep=depth(*dnet, frame);
            cv::imshow(kWinDepth, visualize_depth(dep));
        }

        for (size_t i=0; i<df.size(); i++) {
            const auto d=df[i];

            if (pred) {
                cv::rectangle(frame, d.box, cv::Scalar(0, 0, 255), 2);
                std::string label = cv::format("%d (%.2f)", d.cls, d.score);
                putText(frame, label, cv::Point(d.box.x, d.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
            }

            if (!d.mask.empty()) {
                cv::Mat ma=d.mask, o, mfout, cmask;

                cv::Rect dstRect=d.box;
                cv::Rect dstBounds(0, 0, frame.cols-1, frame.rows-1);
                cv::Rect clipped = dstRect & dstBounds;

                // mask black bg
                cv::Mat maskFrame=cv::Mat::zeros(frame.size(), CV_8U);
                // mask position on black background
                auto mroi=maskFrame(clipped);

                if (bin) {
                    cv::Mat binaryMask;
                    cv::threshold(ma, binaryMask, 0.5, 255, cv::THRESH_BINARY);
                    binaryMask.convertTo(binaryMask, CV_8U);

                    if (contour) {
                        cmask=cv::Mat::zeros(binaryMask.size(), CV_8U);
                        std::vector<std::vector<cv::Point> > cvs;
                        cv::findContours(binaryMask, cvs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

                        for (int ci=0; ci<cvs.size();ci++) {
                            cv::drawContours(cmask, cvs, ci, cv::Scalar(255), -2);
                        }
                        binaryMask=cmask;
                    }

                    if (blur) {
                        cv::blur(binaryMask, binaryMask, cv::Size(9, 9));
                    }
                    cv::resize(binaryMask, binaryMask, d.box.size());
                    binaryMask.copyTo(mroi);
                } else {
                    if (blur) {
                        cv::dilate(ma, ma, cv::Mat());
                        cv::blur(ma, ma, cv::Size(9, 9));
                    }
                    ma.convertTo(ma, CV_8U, 255.0);
                    cv::resize(ma, ma, d.box.size());
                    ma.copyTo(mroi);
                }

                imshow(kWinMask, maskFrame);
            }
            if (!d.pose.empty()) {
                for (size_t pi=0; pi<d.pose.size(); pi++) {
                    auto cp=d.pose[pi];
                    // ignore low points
                    if (cp.z>0.5) {
                        auto p=point3to2(cp);
                        cv::circle(frame, p, 4, cv::Scalar(255, 160, 160), 2);
                        std::string pl = cv::format("%zu (%.2f)", pi, cp.z);
                        putText(frame, pl, p, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);
                    }
                }
                // Nose to eye
                cv::line(frame, point3to2(d.pose[0]), point3to2(d.pose[1]), cv::Scalar(0,120,255), 2);
                cv::line(frame, point3to2(d.pose[1]), point3to2(d.pose[3]), cv::Scalar(0,120,255), 2);

                // Nose to eye
                cv::line(frame, point3to2(d.pose[0]), point3to2(d.pose[2]), cv::Scalar(0,120,255), 2);
                cv::line(frame, point3to2(d.pose[2]), point3to2(d.pose[4]), cv::Scalar(0,120,255), 2);

                // Shoulders
                posel(frame, d.pose, 5, 6, cv::Scalar(60,180,255));

                // hand
                posel(frame, d.pose, 6, 8, cv::Scalar(60,180,255));
                posel(frame, d.pose, 8, 10, cv::Scalar(60,180,255));

                posel(frame, d.pose, 5, 7, cv::Scalar(60,180,255));
                posel(frame, d.pose, 7, 9, cv::Scalar(60,180,255));

                // Feet
                posel(frame, d.pose, 11, 12, cv::Scalar(255,180,255));

                posel(frame, d.pose, 12, 14, cv::Scalar(255,180,255));
                posel(frame, d.pose, 14, 16, cv::Scalar(255,180,255));

                posel(frame, d.pose, 11, 13, cv::Scalar(255,180,255));
                posel(frame, d.pose, 13, 15, cv::Scalar(255,180,255));
            }
        }

        std::string flabel = cv::format("(%.1f) [%d] (%s)", fps, cneti, cname.c_str());
        putText(frame, flabel, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        imshow(kWinMain, frame);

        tm.stop();
        if (f % 16==0) {
            fps=tm.getFPS();
            qDebug() << "FPS: " << fps << tm.getAvgTimeMilli() << frame.size().height << frame.size().width;
        }

        if (writer.isOpened()) {
            writer.write(frame);
        }

        int key = cv::waitKey(1);
        switch (key) {
        case '1':
            set_current_network(0);
            break;
        case '2':
            set_current_network(1);
            break;
        case '3':
            set_current_network(2);
            break;
        case '4':
            set_current_network(3);
            break;
        case '5':
            set_current_network(4);
            break;
        case '6':
            set_current_network(5);
            break;
        case '7':
            set_current_network(6);
            break;
        case '8':
            set_current_network(7);
            break;
        case '9':
            set_current_network(8);
            break;
        case 'q':
            run=false;
            break;
        case 'b':
            bin=!bin;
            break;
        case 'c':
            contour=!contour;
            break;
        case 'm':
            blur=!blur;
            break;
        case 'p':
            pred=!pred;
            break;
        case 'z':
            showdepth=!showdepth;
            break;
        case 'f':
            cv::setWindowProperty(kWinMask, cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN );
            break;
        case 'r':
            cv::setWindowProperty(kWinMask, cv::WND_PROP_FULLSCREEN , cv::WINDOW_NORMAL);
            break;
        case 'd':
            cv::setWindowProperty(kWinMain, cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN );
            break;
        case 'e':
            cv::setWindowProperty(kWinMain, cv::WND_PROP_FULLSCREEN , cv::WINDOW_NORMAL);
            break;
        case ' ':
            cv::waitKey(0);
            break;
        case '+':
            thres+=0.01;
        break;
        case '-':
            thres-=0.01;
            break;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}
