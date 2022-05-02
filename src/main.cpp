#include "transform.hpp"
#include "app.hpp"
//#include "camera.hpp"
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
	auto cameraTransform = Transform();
	cameraTransform.position.z = -2.0f;

	const std::vector<Vertex> vertices = {
		{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
		{{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
		{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}};

	return app.run(
		[&app, &vertices, &cameraTransform] {
			glfwSetWindowUserPointer(app.get_window(), &cameraTransform);

			app.initialize(vertices);
		},
		[&app, &vertices, &cameraTransform] {
			// Handle updates
			if (glfwGetKey(app.get_window(), GLFW_KEY_W) == GLFW_PRESS) {
				cameraTransform.position.z += 0.001;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_S) == GLFW_PRESS) {
				cameraTransform.position.z -= 0.001;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_A) == GLFW_PRESS) {
				cameraTransform.position.x -= 0.001;
			}

			if (glfwGetKey(app.get_window(), GLFW_KEY_D) == GLFW_PRESS) {
				cameraTransform.position.x += 0.001;
			}

			// Handle rendering
			const glm::mat4 projection =
				glm::perspective(60.0f, 1.0f, 0.1f, 200.0f);
			const glm::mat4 view = cameraTransform.get_matrix();
			const glm::mat4 model = glm::mat4(1.0f);

			app.set_transform(projection * view * model);

			app.draw_frame(vertices);
		});
}
