include_directories(
  ${TBB_INCLUDE_DIRS}
  ${ASMJIT_INCLUDE_DIRS}
)

add_executable(mitsuba mitsuba.cpp)

target_link_libraries(mitsuba PRIVATE mitsuba-core mitsuba-render tbb)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|AMD64")
  target_link_libraries(mitsuba PRIVATE asmjit)
endif()

add_dist(mitsuba)

if (APPLE)
  set_target_properties(mitsuba PROPERTIES INSTALL_RPATH "@executable_path")
endif()

if (MSVC)
  set_property(TARGET mitsuba PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)dist")
endif()

include_directories(
  ${OPENEXR_INCLUDE_DIRS}
)

if (MSVC)
  add_definitions(-DOPENEXR_DLL)
endif()

add_executable(normalmap_to_flakes normalmap_to_flakes.cpp)
target_link_libraries(normalmap_to_flakes PRIVATE IlmImf)
add_dist(normalmap_to_flakes)

if (APPLE)
  set_target_properties(normalmap_to_flakes PROPERTIES INSTALL_RPATH "@executable_path")
endif()

if (MSVC)
  set_property(TARGET normalmap_to_flakes PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)dist")
endif()