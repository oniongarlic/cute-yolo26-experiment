cmake_minimum_required(VERSION 3.14)

project(qtopencv LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Core)

if (UNIX)
set(OpenCV_DIR "/opt/OpenCV-5.0.0/lib/cmake/opencv5")
endif (UNIX)

if (MSVC)
set(OpenCV_DIR "C:/Development/opencv-5")
endif (MSVC)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(qtopencv
  main.cpp
)
target_link_libraries(qtopencv Qt6::Core ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS qtopencv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
