#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QFile>
#include <QFileDialog>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);


yolov5_.init(1,0.25,0.45);//如果使用opencv直接打开，就是bgr,则参数为0.此处是qt转换为rgb显示后，直接传递给模型所以是1
}

MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::on_button_detect_clicked()
{



    if(img_.empty()){QMessageBox::question(nullptr, "1请选择图片","提示");return;}
    if(!img_.data){QMessageBox::question(nullptr, "2请选择图片","提示");return; }

    std::vector<Object> objects;
    cv::Mat orig_img=img_;

    yolov5_.inference(orig_img, objects);

    cv::Mat ret= yolov5_.draw_objects(orig_img, objects);

    QMessageBox::question(nullptr,  QString::number(objects.size()),"检测目标个数");

    cv::Mat rgb=ret.clone();

    QImage Img = QImage((const uchar*)(rgb.data), rgb.cols, rgb.rows, rgb.cols * rgb.channels(), QImage::Format_RGB888);


    ui->label->setPixmap(QPixmap::fromImage(Img));

    ui->label->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
    ui->label->setScaledContents(true);
    ui->label->show();

    img_.release();

}

cv::Mat MainWindow::QImage2Mat(QImage & image)
{

    QImage temp = image.copy();
    temp = temp.convertToFormat(QImage::Format_RGB888);
    cv::Mat res(temp.height(),temp.width(),CV_8UC3,(void*)temp.bits(),temp.bytesPerLine());

    return res.clone();

}
void MainWindow::on_button_open_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(
                this, tr("open image file"),
                "./", tr("Image files(*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm *.jpeg);;All files (*.*)"));

    if(fileName.isEmpty())
    {
        QMessageBox mesg;
        mesg.warning(this,"警告","打开图片失败!");
        return;
    }

    QImage img;
    img.load (fileName);

    img_= QImage2Mat(img);

    ui->label->setPixmap(QPixmap::fromImage(img));

    ui->label->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
    ui->label->setScaledContents(true);
    ui->label->show();

}


