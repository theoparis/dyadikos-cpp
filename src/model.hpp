#pragma once
#include "app.hpp"
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace dyadikos::model {
	void processNode(aiNode *node, const aiScene *scene,
					 std::vector<Vertex> &vertices) {
		// process all the node's meshes (if any)
		for (unsigned int i = 0; i < node->mNumMeshes; i++) {
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];

			for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
				const aiFace &face = mesh->mFaces[i];

				const unsigned int idx[3] = {face.mIndices[0], face.mIndices[1],
											 face.mIndices[2]};

				for (int j = 0; j != 3; j++) {
					const aiVector3D v = mesh->mVertices[idx[j]];

					Vertex vertex{};

					vertex.color = glm::vec3(1.0f, 1.0f, 1.0f);
					vertex.pos = glm::vec3(v.x, v.z, v.y);

					vertices.push_back(vertex);
				}
			}
		}
		// then do the same for each of its children
		for (unsigned int i = 0; i < node->mNumChildren; i++) {
			processNode(node->mChildren[i], scene, vertices);
		}
	}

	auto loadModel(const char *path) -> std::vector<Vertex> {
		Assimp::Importer importer;
		auto scene = importer.ReadFile(path, aiProcess_Triangulate);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
			!scene->mRootNode) {
			throw std::runtime_error(importer.GetErrorString());
		}

		std::vector<Vertex> vertices;

		processNode(scene->mRootNode, scene, vertices);

		return vertices;
	}
} // namespace dyadikos::model
