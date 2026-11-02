cmake_minimum_required(VERSION 3.16)
project(aerofusion_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../crates/aerofusion-ffi/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../target/release
)

add_executable(aerofusion_demo main.cpp)

# Target linking against the compiled Rust DLL import library
target_link_libraries(aerofusion_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../target/release/aerofusion_ffi.dll.lib"
)

# Copy the DLL next to the output executable on Windows
add_custom_command(TARGET aerofusion_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../target/release/aerofusion_ffi.dll"
        "$<TARGET_FILE_DIR:aerofusion_demo>"
)
