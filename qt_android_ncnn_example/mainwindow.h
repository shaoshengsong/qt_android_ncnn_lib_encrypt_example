#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "YOLOv5.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    cv::Mat img_;
    YOLOv5 yolov5_;
private slots:
    void on_button_detect_clicked();
    void on_button_open_clicked();
    cv::Mat QImage2Mat(QImage & image);
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
