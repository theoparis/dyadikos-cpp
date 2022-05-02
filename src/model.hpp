#pragma once
#include "App.hpp"
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

namespace dyadikos::model {
	auto loadModel(const char *path) -> std::vector<Vertex> {
		tinygltf::Model model;
		tinygltf::TinyGLTF loader;

		std::string err = 0;
		std::string warn = 0;

		bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);

		if (!warn.empty()) {
			spdlog::warn("Warn: {}\n", warn.c_str());
		}

		if (!err.empty()) {
			spdlog::error("Err: {}\n", err.c_str());
		}

		if (!ret) {
			spdlog::error("Failed to parse glTF\n");
			return {};
		}

		// TODO: process model
		return {};
	}
} // namespace dyadikos::model
