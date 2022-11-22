cmake_minimum_required(VERSION 3.14)
project(yolov5lib LANGUAGES CXX)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(
${CMAKE_SOURCE_DIR}/opencv4.6.0/native/jni/include/opencv2
${CMAKE_SOURCE_DIR}/opencv4.6.0/native/jni/include
${CMAKE_SOURCE_DIR}/ncnn20220420/include
${CMAKE_SOURCE_DIR}/ncnn20220420/include/ncnn

)


set(STATIC_LIBS
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_calib3d.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_gapi.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_objdetect.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_core.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_highgui.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_photo.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_dnn.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_imgcodecs.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_stitching.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_features2d.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_imgproc.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_video.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_flann.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_ml.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/staticlibs/arm64-v8a/libopencv_videoio.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libtbb.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libIlmImf.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibjpeg-turbo.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibtiff.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libtegra_hal.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libade.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibopenjp2.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibwebp.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libcpufeatures.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibpng.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libquirc.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libittnotify.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/liblibprotobuf.a
        ${CMAKE_SOURCE_DIR}/opencv4.6.0/native/3rdparty/libs/arm64-v8a/libtbb.a
        ${CMAKE_SOURCE_DIR}/ncnn20220420/lib/libncnn.a

)


add_library(yolov5lib STATIC
  YOLOv5.cpp
  YOLOv5.h
  yolov5s_6.2.id.h
  yolov5s_6.2.mem.h
)
target_link_libraries(yolov5lib PRIVATE  ${STATIC_LIBS})
target_compile_definitions(yolov5lib  PRIVATE YOLOv5_LIBRARY)




