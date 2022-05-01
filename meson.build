project('dyadikos', 'cpp', default_options : ['cpp_std=c++17'])

if get_option('buildtype').startswith('debug')
  add_project_arguments('-DDEBUG', language : 'cpp')
endif

inc = include_directories('include')

src = [
  'include/app.hpp',
  'include/shader.hpp',
  'include/mesh.hpp',
  'include/transform.hpp',
  'include/primitive.hpp',
  'src/glad.cpp'
]

libdyadikos = static_library(
  'dyadikos',
  sources: src,
  dependencies: [
    dependency('glfw3'),
    dependency('spdlog'),
    dependency('fmt'),
  ],
  install: true,
  include_directories: inc
)
libdyadikos_dep = declare_dependency(
  include_directories: inc,
  link_with: libdyadikos,
  dependencies: [
    dependency('glfw3'),
    dependency('spdlog'),
    dependency('fmt'),
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
