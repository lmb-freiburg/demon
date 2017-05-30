cmake_minimum_required( VERSION 3.5.1 )

project(depthmotionnet)

if( NOT CMAKE_BUILD_TYPE )
        set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE )
endif()

# enable all warnings
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )

add_subdirectory( multivih5datareaderop ) 

