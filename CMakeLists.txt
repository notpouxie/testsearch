# Copyright 2020 Mats Kindahl
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.1)
project(testsearch)
message(STATUS "start running cmake...")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#set(CMAKE_PREFIX_PATH "/home/bviolet/word2vec/libtorch")
set(Torch_DIR "/home/bviolet/word2vec/libtorch/share/cmake/Torch")
find_package(Boost 1.75.0 COMPONENTS system filesystem REQUIRED)
find_package(Torch REQUIRED)
find_package(PostgreSQL REQUIRED)

if (Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})

endif ()

add_executable(${PROJECT_NAME} src/testsearch/testsearch.cpp src/tokenizer/tokenizer.cpp src/tokenizer/unilib/unicode.cpp src/tokenizer/unilib/uninorms.cpp)

if(Boost_FOUND)

    target_link_libraries(testsearch ${Boost_LIBRARIES})

endif()

target_link_libraries(testsearch ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "src/tokenizer/unilib")
target_include_directories(${PROJECT_NAME} PUBLIC "src/tokenizer")

include_directories(${PostgreSQL_INCLUDE_DIRS}
                    ${PostgreSQL_SERVER_INCLUDE_DIRS}
                    ${PostgreSQL_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME} ${PostgreSQL_LIBRARIES})



set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
