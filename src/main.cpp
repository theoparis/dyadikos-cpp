#include "transform.hpp"
#include "app.hpp"
#include "model.hpp"
#include <GLFW/glfw3.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <cerrno>

using namespace dyadikos;

auto get_file_contents(const char *filename) -> std::string {
	std::ifstream in = std::ifstream(filename, std::ios::in | std::ios::binary);
	if (in) {
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();
		return (contents.str());
	}

	throw(errno);
}

auto main() -> int {
	auto app = Application{};
	auto camera_transform = Transform();
	camera_transform.position.z = -2.0f;

	std::vector<Vertex> vertices = {{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
									{{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
									{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}};

	return app.run(
		[&app, &vertices, &camera_transform] {
			glfwSetWindowUserPointer(app.get_window(), &camera_transform);

			vertices = model::loadModel("models/monkey.glb");

			app.initialize(vertices);
		},
		[&app, &vertices, &camera_transform] {
			// Handle updates
			if (glfwGetKey(app.get_window(), GLFW_KEY_W) == GLFW_PRESS) {
				camera_transform.position.z += 0.01;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_S) == GLFW_PRESS) {
				camera_transform.position.z -= 0.01;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_A) == GLFW_PRESS) {
				camera_transform.position.x += 0.01;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_D) == GLFW_PRESS) {
				camera_transform.position.x -= 0.01;
			}

			// Handle rendering
			auto window_size = app.get_window_size();
			const glm::mat4 projection = glm::perspective(
				45.0f, window_size.x / window_size.y, 0.1f, 200.0f);
			const glm::mat4 view = camera_transform.get_matrix();
			const glm::mat4 model =
				glm::mat4(1.0f) * glm::scale(glm::vec3(10.0, 10.0, 10.0));

			app.set_transform(projection * view * model);

			app.draw_frame(vertices);
		});
}
