cmake_minimum_required(VERSION 3.16)
project(aerofusion_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../target/release")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../crates/aerofusion-ffi/include
)

add_executable(aerofusion_demo main.cpp)

if(WIN32)
    target_link_libraries(aerofusion_demo PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../target/release/aerofusion_ffi.dll.lib"
    )
    add_custom_command(TARGET aerofusion_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/../target/release/aerofusion_ffi.dll"
            "$<TARGET_FILE_DIR:aerofusion_demo>"
    )
else()
    target_link_directories(aerofusion_demo PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../target/release"
    )
    target_link_libraries(aerofusion_demo PRIVATE
        aerofusion_ffi
        pthread
        dl
        m
    )
endif()
