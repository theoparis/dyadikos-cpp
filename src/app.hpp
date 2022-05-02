#pragma once
#include <glm/glm.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <functional>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <array>
#include <optional>
#include <set>

namespace dyadikos {
	static void error_callback(int error, const char *description) {
		spdlog::error("Error: {}\n", description);
	}

	const std::vector<const char *> validationLayers = {
		"VK_LAYER_KHRONOS_validation"};

	const std::vector<const char *> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME};

	const int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

	auto CreateDebugUtilsMessengerEXT(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
		const VkAllocationCallbacks *pAllocator,
		VkDebugUtilsMessengerEXT *pCallback) -> VkResult {
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
			instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, pCallback);
		} else {
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void
	DestroyDebugUtilsMessengerEXT(VkInstance instance,
								  VkDebugUtilsMessengerEXT callback,
								  const VkAllocationCallbacks *pAllocator) {
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
			instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr) {
			func(instance, callback, pAllocator);
		}
	}

	struct QueueFamilyIndices {
			std::optional<uint32_t> graphicsFamily;
			std::optional<uint32_t> presentFamily;

			auto isComplete() -> bool {
				return graphicsFamily.has_value() && presentFamily.has_value();
			}
	};

	struct SwapChainSupportDetails {
			vk::SurfaceCapabilitiesKHR capabilities;
			std::vector<vk::SurfaceFormatKHR> formats;
			std::vector<vk::PresentModeKHR> presentModes;
	};

	struct Vertex {
			glm::vec3 pos;
			glm::vec3 color;

			static auto getBindingDescription()
				-> vk::VertexInputBindingDescription {
				vk::VertexInputBindingDescription bindingDescription = {};
				bindingDescription.binding = 0;
				bindingDescription.stride = sizeof(Vertex);
				bindingDescription.inputRate = vk::VertexInputRate::eVertex;

				return bindingDescription;
			}

			static auto getAttributeDescriptions()
				-> std::array<vk::VertexInputAttributeDescription, 2> {
				std::array<vk::VertexInputAttributeDescription, 2>
					attributeDescriptions = {};
				attributeDescriptions[0].binding = 0;
				attributeDescriptions[0].location = 0;
				attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
				attributeDescriptions[0].offset = offsetof(Vertex, pos);

				attributeDescriptions[1].binding = 0;
				attributeDescriptions[1].location = 1;
				attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
				attributeDescriptions[1].offset = offsetof(Vertex, color);

				return attributeDescriptions;
			}
	};

	struct PushConstants {
			glm::mat4 transform;
	};

	class Application {
		private:
			GLFWwindow *window{};
			PushConstants push_constants;

			vk::UniqueInstance instance;
			VkDebugUtilsMessengerEXT callback;
			vk::SurfaceKHR surface;

			vk::PhysicalDevice physical_device;
			vk::UniqueDevice device;

			vk::Queue graphics_queue;
			vk::Queue present_queue;

			vk::SwapchainKHR swap_chain;
			std::vector<vk::Image> swap_chain_images;
			vk::Format swap_chain_image_format;
			vk::Extent2D swap_chain_extent;
			std::vector<vk::ImageView> swap_chain_image_views;
			std::vector<vk::Framebuffer> swap_chain_framebuffers;

			vk::RenderPass render_pass;
			vk::PipelineLayout pipeline_layout;
			vk::Pipeline graphics_pipeline;

			vk::CommandPool command_pool;

			vk::Buffer vertex_buffer;
			vk::DeviceMemory vertex_buffer_memory;

			std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>>
				command_buffers;

			std::vector<vk::Semaphore> image_available_semaphores;
			std::vector<vk::Semaphore> render_finished_semaphores;
			std::vector<vk::Fence> in_flight_fences;
			size_t current_frame = 0;

			bool framebuffer_resized = false;

			glm::vec2 window_size;

			void cleanup_swap_chain() {
				for (auto framebuffer : swap_chain_framebuffers) {
					device->destroyFramebuffer(framebuffer);
				}

				device->freeCommandBuffers(command_pool, command_buffers);

				device->destroyPipeline(graphics_pipeline);
				device->destroyPipelineLayout(pipeline_layout);
				device->destroyRenderPass(render_pass);

				for (auto imageView : swap_chain_image_views) {
					device->destroyImageView(imageView);
				}

				device->destroySwapchainKHR(swap_chain);
			}

			void cleanup() {
				// NOTE: instance destruction is handled by UniqueInstance, same
				// for device

				cleanup_swap_chain();

				device->destroyBuffer(vertex_buffer);
				device->freeMemory(vertex_buffer_memory);

				for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
					device->destroySemaphore(render_finished_semaphores[i]);
					device->destroySemaphore(image_available_semaphores[i]);
					device->destroyFence(in_flight_fences[i]);
				}

				device->destroyCommandPool(command_pool);

				// surface is created by glfw, therefore not using a Unique
				// handle
				instance->destroySurfaceKHR(surface);

				if (enableValidationLayers) {
					DestroyDebugUtilsMessengerEXT(*instance, callback, nullptr);
				}

				glfwDestroyWindow(window);

				glfwTerminate();
			}

			void recreate_swap_chain(const std::vector<Vertex> vertices) {
				int width = 0, height = 0;
				while (width == 0 || height == 0) {
					glfwGetFramebufferSize(window, &width, &height);
					glfwWaitEvents();
				}

				device->waitIdle();

				cleanup_swap_chain();

				create_swap_chain();
				create_image_views();
				create_render_pass();
				create_graphics_pipeline();
				create_frame_buffers();
				create_command_buffers();
				record_command_buffers(vertices);
			}

			void create_instance() {
				if (enableValidationLayers && !checkValidationLayerSupport()) {
					throw std::runtime_error(
						"validation layers requested, but not available!");
				}

				auto appInfo = vk::ApplicationInfo(
					"Dyadikos", VK_MAKE_VERSION(1, 0, 0), "No Engine",
					VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);

				auto extensions = getRequiredExtensions();

				auto createInfo = vk::InstanceCreateInfo(
					vk::InstanceCreateFlags(), &appInfo, 0,
					nullptr, // enabled layers
					static_cast<uint32_t>(extensions.size()),
					extensions.data() // enabled extensions
				);

				if (enableValidationLayers) {
					createInfo.enabledLayerCount =
						static_cast<uint32_t>(validationLayers.size());
					createInfo.ppEnabledLayerNames = validationLayers.data();
				}

				try {
					instance = vk::createInstanceUnique(createInfo, nullptr);
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create instance!");
				}
			}

			void setup_debug_callback() {
				if (!enableValidationLayers) return;

				auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
					vk::DebugUtilsMessengerCreateFlagsEXT(),
					vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
						vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
						vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
					vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
						vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
						vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
					debugCallback, nullptr);

				if (CreateDebugUtilsMessengerEXT(
						*instance,
						reinterpret_cast<
							const VkDebugUtilsMessengerCreateInfoEXT *>(
							&createInfo),
						nullptr, &callback) != VK_SUCCESS) {
					throw std::runtime_error(
						"failed to set up debug callback!");
				}
			}

			void create_surface() {
				VkSurfaceKHR rawSurface = nullptr;
				if (glfwCreateWindowSurface(*instance, window, nullptr,
											&rawSurface) != VK_SUCCESS) {
					throw std::runtime_error(
						"failed to create window surface!");
				}

				surface = rawSurface;
			}

			void pick_physical_device() {
				auto devices = instance->enumeratePhysicalDevices();
				if (devices.size() == 0) {
					throw std::runtime_error(
						"failed to find GPUs with Vulkan support!");
				}

				for (const auto &device : devices) {
					if (isDeviceSuitable(device)) {
						physical_device = device;
						break;
					}
				}

				if (!physical_device) {
					throw std::runtime_error("failed to find a suitable GPU!");
				}
			}

			void create_logical_device() {
				QueueFamilyIndices indices = findQueueFamilies(physical_device);

				std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
				std::set<uint32_t> uniqueQueueFamilies = {
					indices.graphicsFamily.value(),
					indices.presentFamily.value()};

				float queuePriority = 1.0f;

				for (uint32_t queueFamily : uniqueQueueFamilies) {
					queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(),
												  queueFamily,
												  1, // queueCount
												  &queuePriority);
				}

				auto deviceFeatures = vk::PhysicalDeviceFeatures();
				auto createInfo = vk::DeviceCreateInfo(
					vk::DeviceCreateFlags(),
					static_cast<uint32_t>(queueCreateInfos.size()),
					queueCreateInfos.data());
				createInfo.pEnabledFeatures = &deviceFeatures;
				createInfo.enabledExtensionCount =
					static_cast<uint32_t>(deviceExtensions.size());
				createInfo.ppEnabledExtensionNames = deviceExtensions.data();

				if (enableValidationLayers) {
					createInfo.enabledLayerCount =
						static_cast<uint32_t>(validationLayers.size());
					createInfo.ppEnabledLayerNames = validationLayers.data();
				}

				try {
					device = physical_device.createDeviceUnique(createInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to create logical device!");
				}

				graphics_queue =
					device->getQueue(indices.graphicsFamily.value(), 0);
				present_queue =
					device->getQueue(indices.presentFamily.value(), 0);
			}

			void create_swap_chain() {
				SwapChainSupportDetails swapChainSupport =
					querySwapChainSupport(physical_device);

				vk::SurfaceFormatKHR surfaceFormat =
					chooseSwapSurfaceFormat(swapChainSupport.formats);
				vk::PresentModeKHR presentMode =
					chooseSwapPresentMode(swapChainSupport.presentModes);
				vk::Extent2D extent =
					chooseSwapExtent(swapChainSupport.capabilities);

				uint32_t imageCount =
					swapChainSupport.capabilities.minImageCount + 1;
				if (swapChainSupport.capabilities.maxImageCount > 0 &&
					imageCount > swapChainSupport.capabilities.maxImageCount) {
					imageCount = swapChainSupport.capabilities.maxImageCount;
				}

				vk::SwapchainCreateInfoKHR createInfo(
					vk::SwapchainCreateFlagsKHR(), surface, imageCount,
					surfaceFormat.format, surfaceFormat.colorSpace, extent,
					1, // imageArrayLayers
					vk::ImageUsageFlagBits::eColorAttachment);

				QueueFamilyIndices indices = findQueueFamilies(physical_device);
				uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
												 indices.presentFamily.value()};

				if (indices.graphicsFamily != indices.presentFamily) {
					createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
					createInfo.queueFamilyIndexCount = 2;
					createInfo.pQueueFamilyIndices = queueFamilyIndices;
				} else {
					createInfo.imageSharingMode = vk::SharingMode::eExclusive;
				}

				createInfo.preTransform =
					swapChainSupport.capabilities.currentTransform;
				createInfo.compositeAlpha =
					vk::CompositeAlphaFlagBitsKHR::eOpaque;
				createInfo.presentMode = presentMode;
				createInfo.clipped = VK_TRUE;

				createInfo.oldSwapchain = vk::SwapchainKHR(nullptr);

				try {
					swap_chain = device->createSwapchainKHR(createInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create swap chain!");
				}

				swap_chain_images = device->getSwapchainImagesKHR(swap_chain);

				swap_chain_image_format = surfaceFormat.format;
				swap_chain_extent = extent;
			}

			void create_image_views() {
				swap_chain_image_views.resize(swap_chain_images.size());

				for (size_t i = 0; i < swap_chain_images.size(); i++) {
					vk::ImageViewCreateInfo createInfo = {};
					createInfo.image = swap_chain_images[i];
					createInfo.viewType = vk::ImageViewType::e2D;
					createInfo.format = swap_chain_image_format;
					createInfo.components.r = vk::ComponentSwizzle::eIdentity;
					createInfo.components.g = vk::ComponentSwizzle::eIdentity;
					createInfo.components.b = vk::ComponentSwizzle::eIdentity;
					createInfo.components.a = vk::ComponentSwizzle::eIdentity;
					createInfo.subresourceRange.aspectMask =
						vk::ImageAspectFlagBits::eColor;
					createInfo.subresourceRange.baseMipLevel = 0;
					createInfo.subresourceRange.levelCount = 1;
					createInfo.subresourceRange.baseArrayLayer = 0;
					createInfo.subresourceRange.layerCount = 1;

					try {
						swap_chain_image_views[i] =
							device->createImageView(createInfo);
					} catch (vk::SystemError err) {
						throw std::runtime_error(
							"failed to create image views!");
					}
				}
			}

			void create_render_pass() {
				vk::AttachmentDescription colorAttachment = {};
				colorAttachment.format = swap_chain_image_format;
				colorAttachment.samples = vk::SampleCountFlagBits::e1;
				colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
				colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
				colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
				colorAttachment.stencilStoreOp =
					vk::AttachmentStoreOp::eDontCare;
				colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
				colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

				vk::AttachmentReference colorAttachmentRef = {};
				colorAttachmentRef.attachment = 0;
				colorAttachmentRef.layout =
					vk::ImageLayout::eColorAttachmentOptimal;

				vk::SubpassDescription subpass = {};
				subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
				subpass.colorAttachmentCount = 1;
				subpass.pColorAttachments = &colorAttachmentRef;

				vk::SubpassDependency dependency = {};
				dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
				dependency.dstSubpass = 0;
				dependency.srcStageMask =
					vk::PipelineStageFlagBits::eColorAttachmentOutput;
				// dependency.srcAccessMask = 0;
				dependency.dstStageMask =
					vk::PipelineStageFlagBits::eColorAttachmentOutput;
				dependency.dstAccessMask =
					vk::AccessFlagBits::eColorAttachmentRead |
					vk::AccessFlagBits::eColorAttachmentWrite;

				vk::RenderPassCreateInfo renderPassInfo = {};
				renderPassInfo.attachmentCount = 1;
				renderPassInfo.pAttachments = &colorAttachment;
				renderPassInfo.subpassCount = 1;
				renderPassInfo.pSubpasses = &subpass;
				renderPassInfo.dependencyCount = 1;
				renderPassInfo.pDependencies = &dependency;

				try {
					render_pass = device->createRenderPass(renderPassInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create render pass!");
				}
			}

			void create_graphics_pipeline() {
				auto vertShaderCode = readFile("shaders/vert.spv");
				auto fragShaderCode = readFile("shaders/frag.spv");

				auto vertShaderModule = createShaderModule(vertShaderCode);
				auto fragShaderModule = createShaderModule(fragShaderCode);

				vk::PipelineShaderStageCreateInfo shaderStages[] = {
					{vk::PipelineShaderStageCreateFlags(),
					 vk::ShaderStageFlagBits::eVertex, *vertShaderModule,
					 "main"},
					{vk::PipelineShaderStageCreateFlags(),
					 vk::ShaderStageFlagBits::eFragment, *fragShaderModule,
					 "main"}};

				vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
				vertexInputInfo.vertexBindingDescriptionCount = 0;
				vertexInputInfo.vertexAttributeDescriptionCount = 0;

				auto bindingDescription = Vertex::getBindingDescription();
				auto attributeDescriptions = Vertex::getAttributeDescriptions();

				vertexInputInfo.vertexBindingDescriptionCount = 1;
				vertexInputInfo.vertexAttributeDescriptionCount =
					static_cast<uint32_t>(attributeDescriptions.size());
				vertexInputInfo.pVertexBindingDescriptions =
					&bindingDescription;
				vertexInputInfo.pVertexAttributeDescriptions =
					attributeDescriptions.data();

				vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
				inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
				inputAssembly.primitiveRestartEnable = VK_FALSE;

				vk::Viewport viewport = {};
				viewport.x = 0.0f;
				viewport.y = 0.0f;
				viewport.width = (float)swap_chain_extent.width;
				viewport.height = (float)swap_chain_extent.height;
				viewport.minDepth = 0.0f;
				viewport.maxDepth = 1.0f;

				vk::Rect2D scissor = {};
				scissor.offset.x = 0;
				scissor.offset.y = 0;
				scissor.extent = swap_chain_extent;

				vk::PipelineViewportStateCreateInfo viewportState = {};
				viewportState.viewportCount = 1;
				viewportState.pViewports = &viewport;
				viewportState.scissorCount = 1;
				viewportState.pScissors = &scissor;

				vk::PipelineRasterizationStateCreateInfo rasterizer = {};
				rasterizer.depthClampEnable = VK_FALSE;
				rasterizer.rasterizerDiscardEnable = VK_FALSE;
				rasterizer.polygonMode = vk::PolygonMode::eFill;
				rasterizer.lineWidth = 1.0f;
				rasterizer.cullMode = vk::CullModeFlagBits::eBack;
				rasterizer.frontFace = vk::FrontFace::eClockwise;
				rasterizer.depthBiasEnable = VK_FALSE;

				vk::PipelineMultisampleStateCreateInfo multisampling = {};
				multisampling.sampleShadingEnable = VK_FALSE;
				multisampling.rasterizationSamples =
					vk::SampleCountFlagBits::e1;

				vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
				colorBlendAttachment.colorWriteMask =
					vk::ColorComponentFlagBits::eR |
					vk::ColorComponentFlagBits::eG |
					vk::ColorComponentFlagBits::eB |
					vk::ColorComponentFlagBits::eA;
				colorBlendAttachment.blendEnable = VK_FALSE;

				vk::PipelineColorBlendStateCreateInfo colorBlending = {};
				colorBlending.logicOpEnable = VK_FALSE;
				colorBlending.logicOp = vk::LogicOp::eCopy;
				colorBlending.attachmentCount = 1;
				colorBlending.pAttachments = &colorBlendAttachment;
				colorBlending.blendConstants[0] = 0.0f;
				colorBlending.blendConstants[1] = 0.0f;
				colorBlending.blendConstants[2] = 0.0f;
				colorBlending.blendConstants[3] = 0.0f;

				// setup push constants
				vk::PushConstantRange push_constant{
					vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstants)};

				vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
				pipelineLayoutInfo.setLayoutCount = 0;
				pipelineLayoutInfo.pPushConstantRanges = &push_constant;
				pipelineLayoutInfo.pushConstantRangeCount = 1;

				try {
					pipeline_layout =
						device->createPipelineLayout(pipelineLayoutInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to create pipeline layout!");
				}

				vk::GraphicsPipelineCreateInfo pipelineInfo = {};
				pipelineInfo.stageCount = 2;
				pipelineInfo.pStages = shaderStages;
				pipelineInfo.pVertexInputState = &vertexInputInfo;
				pipelineInfo.pInputAssemblyState = &inputAssembly;
				pipelineInfo.pViewportState = &viewportState;
				pipelineInfo.pRasterizationState = &rasterizer;
				pipelineInfo.pMultisampleState = &multisampling;
				pipelineInfo.pColorBlendState = &colorBlending;
				pipelineInfo.layout = pipeline_layout;
				pipelineInfo.renderPass = render_pass;
				pipelineInfo.subpass = 0;
				pipelineInfo.basePipelineHandle = nullptr;

				try {
					graphics_pipeline =
						device->createGraphicsPipeline(nullptr, pipelineInfo)
							.value;
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to create graphics pipeline!");
				}
			}

			void create_frame_buffers() {
				swap_chain_framebuffers.resize(swap_chain_image_views.size());

				for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
					vk::ImageView attachments[] = {swap_chain_image_views[i]};

					vk::FramebufferCreateInfo framebufferInfo = {};
					framebufferInfo.renderPass = render_pass;
					framebufferInfo.attachmentCount = 1;
					framebufferInfo.pAttachments = attachments;
					framebufferInfo.width = swap_chain_extent.width;
					framebufferInfo.height = swap_chain_extent.height;
					framebufferInfo.layers = 1;

					try {
						swap_chain_framebuffers[i] =
							device->createFramebuffer(framebufferInfo);
					} catch (vk::SystemError err) {
						throw std::runtime_error(
							"failed to create framebuffer!");
					}
				}
			}
			void create_command_pool() {
				QueueFamilyIndices queueFamilyIndices =
					findQueueFamilies(physical_device);

				vk::CommandPoolCreateInfo poolInfo = {};
				poolInfo.queueFamilyIndex =
					queueFamilyIndices.graphicsFamily.value();
				poolInfo.setFlags(
					vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

				try {
					command_pool = device->createCommandPool(poolInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create command pool!");
				}
			}

			void create_vertex_buffer(const std::vector<Vertex> &vertices) {
				vk::DeviceSize bufferSize =
					sizeof(vertices[0]) * vertices.size();

				vk::Buffer stagingBuffer;
				vk::DeviceMemory stagingBufferMemory;
				createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
							 vk::MemoryPropertyFlagBits::eHostVisible |
								 vk::MemoryPropertyFlagBits::eHostCoherent,
							 stagingBuffer, stagingBufferMemory);

				void *data =
					device->mapMemory(stagingBufferMemory, 0, bufferSize);
				memcpy(data, vertices.data(), (size_t)bufferSize);
				device->unmapMemory(stagingBufferMemory);

				createBuffer(bufferSize,
							 vk::BufferUsageFlagBits::eTransferDst |
								 vk::BufferUsageFlagBits::eVertexBuffer,
							 vk::MemoryPropertyFlagBits::eDeviceLocal,
							 vertex_buffer, vertex_buffer_memory);

				copyBuffer(stagingBuffer, vertex_buffer, bufferSize);

				device->destroyBuffer(stagingBuffer);
				device->freeMemory(stagingBufferMemory);
			}

			void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
							  vk::MemoryPropertyFlags properties,
							  vk::Buffer &buffer,
							  vk::DeviceMemory &bufferMemory) {
				vk::BufferCreateInfo bufferInfo = {};
				bufferInfo.size = size;
				bufferInfo.usage = usage;
				bufferInfo.sharingMode = vk::SharingMode::eExclusive;

				try {
					buffer = device->createBuffer(bufferInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create buffer!");
				}

				vk::MemoryRequirements memRequirements =
					device->getBufferMemoryRequirements(buffer);

				vk::MemoryAllocateInfo allocInfo = {};
				allocInfo.allocationSize = memRequirements.size;
				allocInfo.memoryTypeIndex =
					findMemoryType(memRequirements.memoryTypeBits, properties);

				try {
					bufferMemory = device->allocateMemory(allocInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to allocate buffer memory!");
				}

				device->bindBufferMemory(buffer, bufferMemory, 0);
			}

			void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
							VkDeviceSize size) {
				vk::CommandBufferAllocateInfo allocInfo = {};
				allocInfo.level = vk::CommandBufferLevel::ePrimary;
				allocInfo.commandPool = command_pool;
				allocInfo.commandBufferCount = 1;

				vk::CommandBuffer commandBuffer =
					device->allocateCommandBuffers(allocInfo)[0];

				vk::CommandBufferBeginInfo beginInfo = {};
				beginInfo.flags =
					vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

				commandBuffer.begin(beginInfo);

				vk::BufferCopy copyRegion = {};
				copyRegion.srcOffset = 0; // Optional
				copyRegion.dstOffset = 0; // Optional
				copyRegion.size = size;
				commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

				commandBuffer.end();

				vk::SubmitInfo submitInfo = {};
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &commandBuffer;

				graphics_queue.submit(submitInfo, nullptr);
				graphics_queue.waitIdle();

				device->freeCommandBuffers(command_pool, commandBuffer);
			}

			auto findMemoryType(uint32_t typeFilter,
								vk::MemoryPropertyFlags properties)
				-> uint32_t {
				vk::PhysicalDeviceMemoryProperties memProperties =
					physical_device.getMemoryProperties();

				for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
					if ((typeFilter & (1 << i)) &&
						(memProperties.memoryTypes[i].propertyFlags &
						 properties) == properties) {
						return i;
					}
				}

				throw std::runtime_error(
					"failed to find suitable memory type!");
			}

			void create_command_buffers() {
				command_buffers.resize(swap_chain_framebuffers.size());

				vk::CommandBufferAllocateInfo allocInfo = {};
				allocInfo.commandPool = command_pool;
				allocInfo.level = vk::CommandBufferLevel::ePrimary;
				allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

				try {
					command_buffers = device->allocateCommandBuffers(allocInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to allocate command buffers!");
				}
			}

			void record_command_buffers(const std::vector<Vertex> &vertices) {
				for (size_t i = 0; i < command_buffers.size(); i++) {
					record_command_buffer(vertices, i);
				}
			}

			void record_command_buffer(const std::vector<Vertex> &vertices,
									   int i) {
				vk::CommandBufferBeginInfo beginInfo = {};
				beginInfo.flags =
					vk::CommandBufferUsageFlagBits::eSimultaneousUse;

				try {
					command_buffers[i].begin(beginInfo);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to begin recording command buffer!");
				}

				vk::RenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.renderPass = render_pass;
				renderPassInfo.framebuffer = swap_chain_framebuffers[i];
				renderPassInfo.renderArea.offset.x = 0;
				renderPassInfo.renderArea.offset.y = 0;
				renderPassInfo.renderArea.extent = swap_chain_extent;

				vk::ClearValue clearColor = {
					std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
				renderPassInfo.clearValueCount = 1;
				renderPassInfo.pClearValues = &clearColor;

				command_buffers[i].beginRenderPass(
					renderPassInfo, vk::SubpassContents::eInline);

				command_buffers[i].bindPipeline(
					vk::PipelineBindPoint::eGraphics, graphics_pipeline);

				vk::Buffer vertexBuffers[] = {vertex_buffer};
				vk::DeviceSize offsets[] = {0};
				command_buffers[i].bindVertexBuffers(0, 1, vertexBuffers,
													 offsets);

				command_buffers[i].pushConstants(
					pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0,
					sizeof(PushConstants), &push_constants);

				command_buffers[i].draw(static_cast<uint32_t>(vertices.size()),
										1, 0, 0);

				command_buffers[i].endRenderPass();

				try {
					command_buffers[i].end();
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to record command buffer!");
				}
			}

			void create_sync_objects() {
				image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
				render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
				in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

				try {
					for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
						image_available_semaphores[i] =
							device->createSemaphore({});
						render_finished_semaphores[i] =
							device->createSemaphore({});
						in_flight_fences[i] = device->createFence(
							{vk::FenceCreateFlagBits::eSignaled});
					}
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create synchronization "
											 "objects for a frame!");
				}
			}

			auto createShaderModule(const std::vector<char> &code)
				-> vk::UniqueShaderModule {
				try {
					return device->createShaderModuleUnique(
						{vk::ShaderModuleCreateFlags(), code.size(),
						 reinterpret_cast<const uint32_t *>(code.data())});
				} catch (vk::SystemError err) {
					throw std::runtime_error("failed to create shader module!");
				}
			}

			auto chooseSwapSurfaceFormat(
				const std::vector<vk::SurfaceFormatKHR> &availableFormats)
				-> vk::SurfaceFormatKHR {
				if (availableFormats.size() == 1 &&
					availableFormats[0].format == vk::Format::eUndefined) {
					return {vk::Format::eB8G8R8A8Unorm,
							vk::ColorSpaceKHR::eSrgbNonlinear};
				}

				for (const auto &availableFormat : availableFormats) {
					if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
						availableFormat.colorSpace ==
							vk::ColorSpaceKHR::eSrgbNonlinear) {
						return availableFormat;
					}
				}

				return availableFormats[0];
			}

			auto chooseSwapPresentMode(
				const std::vector<vk::PresentModeKHR> availablePresentModes)
				-> vk::PresentModeKHR {
				vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

				for (const auto &availablePresentMode : availablePresentModes) {
					if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
						return availablePresentMode;
					} else if (availablePresentMode ==
							   vk::PresentModeKHR::eImmediate) {
						bestMode = availablePresentMode;
					}
				}

				return bestMode;
			}

			auto
			chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
				-> vk::Extent2D {
				if (capabilities.currentExtent.width !=
					std::numeric_limits<uint32_t>::max()) {
					return capabilities.currentExtent;
				} else {
					int width = 0, height = 0;
					glfwGetFramebufferSize(window, &width, &height);

					vk::Extent2D actualExtent = {static_cast<uint32_t>(width),
												 static_cast<uint32_t>(height)};

					actualExtent.width =
						std::max(capabilities.minImageExtent.width,
								 std::min(capabilities.maxImageExtent.width,
										  actualExtent.width));
					actualExtent.height =
						std::max(capabilities.minImageExtent.height,
								 std::min(capabilities.maxImageExtent.height,
										  actualExtent.height));

					return actualExtent;
				}
			}

			auto querySwapChainSupport(const vk::PhysicalDevice &device)
				-> SwapChainSupportDetails {
				SwapChainSupportDetails details;
				details.capabilities =
					device.getSurfaceCapabilitiesKHR(surface);
				details.formats = device.getSurfaceFormatsKHR(surface);
				details.presentModes =
					device.getSurfacePresentModesKHR(surface);

				return details;
			}

			auto isDeviceSuitable(const vk::PhysicalDevice &device) -> bool {
				QueueFamilyIndices indices = findQueueFamilies(device);

				bool extensionsSupported = checkDeviceExtensionSupport(device);

				bool swapChainAdequate = false;
				if (extensionsSupported) {
					SwapChainSupportDetails swapChainSupport =
						querySwapChainSupport(device);
					swapChainAdequate = !swapChainSupport.formats.empty() &&
										!swapChainSupport.presentModes.empty();
				}

				return indices.isComplete() && extensionsSupported &&
					   swapChainAdequate;
			}

			auto checkDeviceExtensionSupport(const vk::PhysicalDevice &device)
				-> bool {
				std::set<std::string> requiredExtensions(
					deviceExtensions.begin(), deviceExtensions.end());

				for (const auto &extension :
					 device.enumerateDeviceExtensionProperties()) {
					requiredExtensions.erase(extension.extensionName);
				}

				return requiredExtensions.empty();
			}

			auto findQueueFamilies(vk::PhysicalDevice device)
				-> QueueFamilyIndices {
				QueueFamilyIndices indices;

				auto queueFamilies = device.getQueueFamilyProperties();

				int i = 0;
				for (const auto &queueFamily : queueFamilies) {
					if (queueFamily.queueCount > 0 &&
						queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
						indices.graphicsFamily = i;
					}

					if (queueFamily.queueCount > 0 &&
						device.getSurfaceSupportKHR(i, surface)) {
						indices.presentFamily = i;
					}

					if (indices.isComplete()) {
						break;
					}

					i++;
				}

				return indices;
			}

			auto getRequiredExtensions() -> std::vector<const char *> {
				uint32_t glfwExtensionCount = 0;
				const char **glfwExtensions = nullptr;
				glfwExtensions =
					glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

				std::vector<const char *> extensions(
					glfwExtensions, glfwExtensions + glfwExtensionCount);

				if (enableValidationLayers) {
					extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
				}

				return extensions;
			}

			auto checkValidationLayerSupport() -> bool {
				auto availableLayers = vk::enumerateInstanceLayerProperties();
				for (const char *layerName : validationLayers) {
					bool layerFound = false;

					for (const auto &layerProperties : availableLayers) {
						if (strcmp(layerName, layerProperties.layerName) == 0) {
							layerFound = true;
							break;
						}
					}

					if (!layerFound) {
						return false;
					}
				}

				return true;
			}

			static auto readFile(const std::string &filename)
				-> std::vector<char> {
				std::ifstream file(filename, std::ios::ate | std::ios::binary);

				if (!file.is_open()) {
					throw std::runtime_error("failed to open file!");
				}

				size_t fileSize = (size_t)file.tellg();
				std::vector<char> buffer(fileSize);

				file.seekg(0);
				file.read(buffer.data(), fileSize);

				file.close();

				return buffer;
			}

			static VKAPI_ATTR auto VKAPI_CALL debugCallback(
				VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
				VkDebugUtilsMessageTypeFlagsEXT messageType,
				const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
				void *pUserData) -> VkBool32 {
				spdlog::error("validation layer: {}", pCallbackData->pMessage);

				return VK_FALSE;
			}

		public:
			auto get_window_size() -> glm::vec2 { return window_size; }

			void initialize(const std::vector<Vertex> &vertices) {
				create_instance();
				setup_debug_callback();
				create_surface();
				pick_physical_device();
				create_logical_device();
				create_swap_chain();
				create_image_views();
				create_render_pass();
				create_graphics_pipeline();
				create_frame_buffers();
				create_command_pool();
				create_vertex_buffer(vertices);
				create_command_buffers();
				record_command_buffers(vertices);
				create_sync_objects();
			}

			void draw_frame(const std::vector<Vertex> &vertices) {
				device->waitForFences(1, &in_flight_fences[current_frame],
									  VK_TRUE,
									  std::numeric_limits<uint64_t>::max());

				uint32_t image_index = 0;
				try {
					vk::ResultValue result = device->acquireNextImageKHR(
						swap_chain, std::numeric_limits<uint64_t>::max(),
						image_available_semaphores[current_frame], nullptr);
					image_index = result.value;
				} catch (vk::OutOfDateKHRError err) {
					recreate_swap_chain(vertices);
					return;
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to acquire swap chain image!");
				}

				command_buffers[image_index].reset(
					vk::CommandBufferResetFlags(0));

				record_command_buffer(vertices, image_index);

				vk::SubmitInfo submit_info = {};

				vk::Semaphore wait_semaphores[] = {
					image_available_semaphores[current_frame]};
				vk::PipelineStageFlags wait_stages[] = {
					vk::PipelineStageFlagBits::eColorAttachmentOutput};
				submit_info.waitSemaphoreCount = 1;
				submit_info.pWaitSemaphores = wait_semaphores;
				submit_info.pWaitDstStageMask = wait_stages;

				submit_info.commandBufferCount = 1;
				submit_info.pCommandBuffers = &command_buffers[image_index];

				vk::Semaphore signal_semaphores[] = {
					render_finished_semaphores[current_frame]};
				submit_info.signalSemaphoreCount = 1;
				submit_info.pSignalSemaphores = signal_semaphores;

				device->resetFences(1, &in_flight_fences[current_frame]);

				try {
					graphics_queue.submit(submit_info,
										  in_flight_fences[current_frame]);
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to submit draw command buffer!");
				}

				vk::PresentInfoKHR present_info = {};
				present_info.waitSemaphoreCount = 1;
				present_info.pWaitSemaphores = signal_semaphores;

				vk::SwapchainKHR swap_chains[] = {swap_chain};
				present_info.swapchainCount = 1;
				present_info.pSwapchains = swap_chains;
				present_info.pImageIndices = &image_index;

				vk::Result result_present;
				try {
					result_present = present_queue.presentKHR(present_info);
				} catch (vk::OutOfDateKHRError err) {
					result_present = vk::Result::eErrorOutOfDateKHR;
				} catch (vk::SystemError err) {
					throw std::runtime_error(
						"failed to present swap chain image!");
				}

				if (result_present == vk::Result::eSuboptimalKHR ||
					result_present == vk::Result::eSuboptimalKHR ||
					framebuffer_resized) {
					framebuffer_resized = false;
					recreate_swap_chain(vertices);
					return;
				}

				current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
			}

			auto get_window() -> GLFWwindow * { return window; }

			auto set_transform(const glm::mat4 &new_transform) {
				push_constants.transform = new_transform;
			}

			auto run(std::function<void()> initFn, std::function<void()> loopFn)
				-> int {
				glfwSetErrorCallback(error_callback);

				if (!glfwInit()) exit(EXIT_FAILURE);

				glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
				glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
				glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

				window =
					glfwCreateWindow(640, 480, "Dyadikos", nullptr, nullptr);
				if (!window) {
					glfwTerminate();
					return -1;
				}

				initFn();

				while (!glfwWindowShouldClose(window)) {
					int width = 0;
					int height = 0;

					glfwGetFramebufferSize(window, &width, &height);
					// auto ratio = (float)width / (float)height;
					window_size = glm::vec2(width, height);

					loopFn();

					glfwPollEvents();
				}

				device->waitIdle();
				cleanup();

				return 0;
			}
	};

} // namespace dyadikos
