cmake_minimum_required(VERSION 3.16)
project(aerofusion_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../target/release")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(AEROFUSION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(AEROFUSION_MANIFEST "${AEROFUSION_ROOT}/Cargo.toml")

add_custom_target(aerofusion_rust_release
    COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path "${AEROFUSION_MANIFEST}"
    WORKING_DIRECTORY "${AEROFUSION_ROOT}"
    USES_TERMINAL
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../crates/aerofusion-ffi/include
)

add_executable(aerofusion_demo main.cpp)
add_dependencies(aerofusion_demo aerofusion_rust_release)

if(WIN32)
    target_link_libraries(aerofusion_demo PRIVATE
        "${AEROFUSION_ROOT}/target/release/aerofusion_ffi.dll.lib"
    )
    add_custom_command(TARGET aerofusion_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${AEROFUSION_ROOT}/target/release/aerofusion_ffi.dll"
            "$<TARGET_FILE_DIR:aerofusion_demo>"
    )
else()
    target_link_directories(aerofusion_demo PRIVATE
        "${AEROFUSION_ROOT}/target/release"
    )
    target_link_libraries(aerofusion_demo PRIVATE
        aerofusion_ffi
        pthread
        dl
        m
    )
endif()
