QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
DEFINES += QT_DEPRECATED_WARNINGS
# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    YOLOv5.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

android {
ANDROID_OPENCV = D:/test/example/opencv4.6.0/native
ANDROID_NCNN = D:/test/example/

INCLUDEPATH += \
$$ANDROID_OPENCV/jni/include/opencv2 \
$$ANDROID_OPENCV/jni/include \
$$ANDROID_NCNN/ncnn20220420/include \
$$ANDROID_NCNN/ncnn20220420/include/ncnn


LIBS += \
$$ANDROID_NCNN/libyolov5lib.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_ml.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_objdetect.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_calib3d.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_video.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_features2d.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_highgui.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_flann.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_imgproc.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_dnn.a \
$$ANDROID_OPENCV/staticlibs/arm64-v8a/libopencv_core.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libcpufeatures.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libIlmImf.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/liblibjpeg-turbo.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/liblibpng.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/liblibprotobuf.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/liblibtiff.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/liblibwebp.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libquirc.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libtbb.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libtegra_hal.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libade.a \
$$ANDROID_OPENCV/3rdparty/libs/arm64-v8a/libittnotify.a \
$$ANDROID_OPENCV/libs/arm64-v8a/libopencv_java4.so \
$$ANDROID_NCNN/ncnn20220420/lib/libncnn.a \
C:/Microsoft/AndroidNDK/android-ndk-r21e/platforms/android-28/arch-arm64/usr/lib/libandroid.so \
C:/Microsoft/AndroidNDK/android-ndk-r21e/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/9.0.9/lib/linux/aarch64/libomp.a \
C:/Microsoft/AndroidNDK/android-ndk-r21e/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/9.0.9/lib/linux/aarch64/libomp.so

}

ANDROID_EXTRA_LIBS +=$$ANDROID_OPENCV/libs/arm64-v8a/libopencv_java4.so \
                      C:/Microsoft/AndroidNDK/android-ndk-r21e/platforms/android-28/arch-arm64/usr/lib/libandroid.so \
                     C:/Microsoft/AndroidNDK/android-ndk-r21e/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/9.0.9/lib/linux/aarch64/libomp.so
