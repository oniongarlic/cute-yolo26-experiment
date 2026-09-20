cmake_minimum_required(VERSION 3.14)

project(qtopencv LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

#set(OpenCV_DIR "C:/Qt/opencv/build")
set(OpenCV_DIR "C:/Development/opencv-5")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(qtopencv
  main.cpp
)
target_link_libraries(qtopencv Qt${QT_VERSION_MAJOR}::Core ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS qtopencv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
