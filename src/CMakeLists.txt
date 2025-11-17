add_subdirectory(energy)

target_sources(${PROJECT_NAME} PRIVATE
					gco_shape_matching.hpp
					gco_shape_matching_pointwise.cpp
					gco_shape_matching_trianglewise.cpp
					gco_shape_matching_trianglewise_init.cpp
					gco_shape_matching_trianglewise_algorithm.cpp
					gco_shape_matching_trianglewise_helpers.cpp
					gco_shape_matching.cpp)