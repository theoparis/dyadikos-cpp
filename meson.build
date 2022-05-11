project('dyadikos', 'cpp', default_options : ['cpp_std=c++17'])

if get_option('buildtype').startswith('debug')
  add_project_arguments('-DDEBUG', language : 'cpp')
endif

inc = include_directories('src')

src = [
  'src/app.hpp',
  'src/transform.hpp',
  'src/camera.hpp',
  'src/model.hpp',
]

# dependencies
dep_glfw = dependency('glfw3')
dep_spdlog = dependency('spdlog')
dep_fmt = dependency('fmt')
dep_vk = dependency('vulkan')
dep_assimp = dependency('assimp')

libdyadikos = static_library(
  'dyadikos',
  sources: src,
  dependencies: [
    dep_glfw,
    dep_spdlog,
    dep_fmt,
    dep_vk,
    dep_assimp,
  ],
  install: true,
  include_directories: inc
)
libdyadikos_dep = declare_dependency(
  include_directories: inc,
  link_with: libdyadikos,
  dependencies: [
    dep_glfw,
    dep_spdlog,
    dep_fmt,
    dep_vk,
    dep_assimp,
  ],
)

executable(
  'dyadikos',
  sources: ['src/main.cpp'],
  dependencies: [
    libdyadikos_dep
  ],
  install: true,
)
