# ariadne-release-config.cmake
# Auto-detected after extracting a pre-built release archive.
if(NOT TARGET ariadne::ariadne)
  add_library(ariadne::ariadne STATIC IMPORTED)
  get_filename_component(_d "${CMAKE_CURRENT_LIST_DIR}" PATH)

  if(WIN32)
    set_target_properties(ariadne::ariadne PROPERTIES
      IMPORTED_LOCATION "${_d}/lib/ariadne.lib"
      INTERFACE_INCLUDE_DIRECTORIES "${_d}/include")
  else()
    set_target_properties(ariadne::ariadne PROPERTIES
      IMPORTED_LOCATION "${_d}/lib/libariadne.a"
      INTERFACE_INCLUDE_DIRECTORIES "${_d}/include")
  endif()
endif()
