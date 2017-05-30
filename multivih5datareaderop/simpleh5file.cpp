//
//  DeMoN - Depth Motion Network
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "simpleh5file.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <mutex>

#define SIMPLEH5FILE_MAXIMUM_NUMBER_OF_DIMENSIONS 32


#define _TOSTRING_(x) #x
#define TOSTRING(x) _TOSTRING_(x)

#define SIMPLEH5FILE_CHECK_OPEN                                               \
{                                                                             \
if( !is_open )                                                                \
  throw std::runtime_error(__FILE__ ":" TOSTRING(__LINE__)                    \
                           ": File is not open");                             \
}                                                                             

#define SIMPLEH5FILE_CHECK_OPEN_AND_WRITE                                     \
{                                                                             \
if( !is_open )                                                                \
  throw std::runtime_error(__FILE__ ":" TOSTRING(__LINE__)                     \
                           ": File is not open");                             \
if( mode == READ )                                                            \
  throw std::runtime_error(__FILE__ ":" TOSTRING(__LINE__)": Cannot write! "  \
                           "File is opened in read only mode");               \
}                                                                             


#define LOCK_GUARD LockGuard<std::recursive_mutex> lock(global_SimpleH5File_mutex,use_locking);

namespace
{
  std::recursive_mutex global_SimpleH5File_mutex;

  template <class T>
  class LockGuard
  {
  public:
    LockGuard(T& m, bool enabled)
      :m(m), enabled(enabled)
    {
      if( enabled )
        m.lock();
    }

    LockGuard(const LockGuard<T>&) = delete;
    LockGuard<T>& operator=(const LockGuard<T>&) = delete;

    ~LockGuard()
    {
      if( enabled )
        m.unlock();
    }
  private:
    T& m;
    bool enabled;
  };

  class HId
  {
  public:
    HId():id(-1){}
    HId(hid_t _id):id(_id){
      //std::cerr << "ctor HId" << this << "\n";
    }
    hid_t id;
    ~HId()
    {
      //std::cout << "closed\n";
      herr_t status = close();
      if( status < 0 )
        throw std::runtime_error("Close failed");
    }
    HId& operator=( const hid_t& _id )
    {
      id = _id;
      return *this;
    }

    herr_t close()
    {
      if( id < 0 )
        return -1;
      H5I_type_t type = H5Iget_type(id);
      herr_t status = -1;
      switch( type )
      {
      case H5I_FILE:
        //std::cerr << "closing file " << id << "  " << this << "\n";
        status = H5Fclose(id);
        break;
      //case H5I_GROUP:
        //status = H5Gclose(id);
        //break;
      //case H5I_DATATYPE:
        //status = H5Tclose(id);
        //break;
      //case H5I_DATASET:
        //status = H5Dclose(id);
        //break;
      case H5I_GROUP:
        //std::cerr << "close group " << type << "  " << this << "\n";
      case H5I_DATATYPE:
        //std::cerr << "close datatype " << type << "  " << this << "\n";
      case H5I_DATASET:
        //std::cerr << "close dataset " << type << "  " << this << "\n";
        status = H5Oclose(id);
        break;
      case H5I_DATASPACE:
        //std::cerr << "close dataspace " << type << "  " << this << "\n";
        status = H5Sclose(id);
        break;
      case H5I_ATTR:
        //std::cerr << "close attribute " << type << "  " << this << "\n";
        status = H5Aclose(id);
        break;
      default:
        //std::cerr << "close unknown " << type << "  " << this << "\n";
        break;
      }
      return status;
    }

  private:
    HId( const HId& other )
    {
      throw std::runtime_error("copy ctor not allowed for HId");
    }
  };

  /*!
   *  Struct for converting native types to hdf5 types
   */
  template <class T>
  struct nativeToH5Type
  {
    static hid_t type() 
    {
      throw std::runtime_error("Unknown type");
      return -1;
    }
  };

  template <>
  struct nativeToH5Type<char>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_CHAR);
    }
  };

  template <>
  struct nativeToH5Type<unsigned char>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_UCHAR);
    }
  };

  template <>
  struct nativeToH5Type<short>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_SHORT);
    }
  };

  template <>
  struct nativeToH5Type<unsigned short>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_USHORT);
    }
  };

  template <>
  struct nativeToH5Type<int>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_INT);
    }
  };

  template <>
  struct nativeToH5Type<unsigned int>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_UINT);
    }
  };

  template <>
  struct nativeToH5Type<long>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_LONG);
    }
  };

  template <>
  struct nativeToH5Type<unsigned long>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_ULONG);
    }
  };

  template <>
  struct nativeToH5Type<float>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_FLOAT);
    }
  };

  template <>
  struct nativeToH5Type<double>
  {
    static hid_t type() 
    {
      return H5Tcopy(H5T_NATIVE_DOUBLE);
    }
  };

  // function for debugging h5 types
  void checkH5TypeEqual( hid_t type )
  {
    std::cout << std::endl << std::endl << std::endl;
    hid_t tmp;
    htri_t tri;
    tmp = nativeToH5Type<char>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " char " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<unsigned char>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " unsigned char " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<short>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " short " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<unsigned short>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " unsigned short " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<float>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " float " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<double>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " double " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<unsigned int>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " unsigned int " << tri << std::endl;
    H5Tclose(tmp);

    tmp = nativeToH5Type<int>::type();
    tri = H5Tequal(type,tmp);
    std::cout << " int " << tri << std::endl;
    H5Tclose(tmp);
  }

  /*!
   *  Returns the path to the parent object.
   *
   *  \param path  E.g. '/group/dataset'
   *  \return E.g. '/group'
   */
  std::string getParentPath( const std::string& path )
  {
    size_t pos = path.find_last_of( '/' );
    return path.substr(0,pos);
  }


  /*!
   *  Returns the name of the object
   *
   *  \param path  E.g. '/group/dataset'
   *  \return E.g. 'dataset'
   */
  std::string getObjectName( const std::string& path )
  {
    size_t pos = path.find_last_of( '/' );
    return path.substr(pos+1);
  }


  /*!
   *  Converts a path to vector containing the elements of the path.
   *  E.g. '/group/gr2/bla/dset' is converted to ['group','gr2','bla','dset']
   */
  std::vector<std::string> pathToVector( const std::string& path )
  {
    std::vector<std::string> result;

    std::string tmp( path );
    // remove possible starting '/'
    if( tmp[0] == '/' )
      tmp.erase(0,1);

    size_t pos = 0;
    while( pos != std::string::npos )
    {
      size_t new_pos = tmp.find_first_of('/', pos );
      result.push_back( tmp.substr(pos,new_pos-pos) );
      if( new_pos == std::string::npos )
        break;
      else
        pos = new_pos+1;
    }
    return result;
  }
  
}



SimpleH5File::SimpleH5File( bool use_locking )
  :mode(READ),file_id(-1),is_open(false), use_locking(use_locking)
{ }


SimpleH5File::SimpleH5File( const std::string& filename, FileMode _mode, bool use_locking )
  :mode(READ),file_id(-1),is_open(false), use_locking(use_locking)
{
  open( filename, _mode );
}


SimpleH5File::~SimpleH5File()
{
  close();
}


void SimpleH5File::open( const std::string& filename, FileMode _mode )
{
  LOCK_GUARD
  // close the currently opened file
  if( is_open )
  {
    close();
  }

  mode = _mode;
  
  // if the file does not exist use truncate
  if( mode == READ_WRITE && !std::ifstream(filename.c_str()) )
    mode = TRUNCATE;

  switch( mode )
  {
  case TRUNCATE:
    file_id = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    if( file_id < 0 )
      throw std::runtime_error("Cannot create or replace file " + filename);
    break;
  case READ:
    file_id = H5Fopen( filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
    if( file_id < 0 )
      throw std::runtime_error("Cannot open file " + filename + " in read only mode");
    break;
  case READ_WRITE:
    file_id = H5Fopen( filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
    if( file_id < 0 )
      throw std::runtime_error("Cannot open file " + filename + " in read write mode");
    break;
  }
  if( file_id >= 0 )
    is_open = true;
}


bool SimpleH5File::isOpen() const
{
  LOCK_GUARD
  return is_open;
}


bool SimpleH5File::useLocking() const
{
  return use_locking;
}


void SimpleH5File::close()
{
  LOCK_GUARD
  if( !is_open )
    return;

  herr_t status = H5Fclose( file_id );
  if( status < 0 ) 
    throw std::runtime_error("Closing file failed");
  else
    is_open = false;
}


void SimpleH5File::makeGroup( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE;

  if(exists(path)) return;

  hid_t lcpl_id = H5Pcreate( H5P_LINK_CREATE );
  if( lcpl_id == -1 )
  {
    throw std::runtime_error("Creating link creation property list failed");
  }
  herr_t status = H5Pset_create_intermediate_group( lcpl_id, 1 );
  if( status < 0 )
  {
    H5Pclose(lcpl_id);
    throw std::runtime_error("H5Pset_create_intermediate_group failed");
  }

  std::string tmp( simplifyPath(path) );
  
  HId group( H5Gcreate2(file_id, tmp.c_str(), lcpl_id, H5P_DEFAULT, H5P_DEFAULT) );
  if( group.id < 0 )
  {
    throw std::runtime_error("Creating of group failed");
  }
  
  status = H5Pclose( lcpl_id );
  if( status < 0 )
  {
    throw std::runtime_error("Closing link creation property list failed");
  }
}




void SimpleH5File::remove( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE;

  std::string tmp( simplifyPath(path) );
  herr_t status = H5Ldelete(file_id, tmp.c_str(), H5P_DEFAULT);
  if( status < 0 )
  {
    throw std::runtime_error("Deleting " + path + " failed");
  }
}




bool SimpleH5File::isGroup( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string tmp( simplifyPath(path) );
  H5O_info_t info;
  herr_t status = H5Oget_info_by_name(file_id, tmp.c_str(), &info, H5P_DEFAULT);

  if( status >= 0 )
  {
    if( info.type == H5O_TYPE_GROUP )
      return true;
  }
  return false;
}




bool SimpleH5File::isDataset( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string tmp( simplifyPath(path) );
  H5O_info_t info;
  herr_t status = H5Oget_info_by_name(file_id, tmp.c_str(), &info, H5P_DEFAULT);

  if( status >= 0 )
  {
    if( info.type == H5O_TYPE_DATASET )
      return true;
  }

  return false;

}


template <class NATIVE_TYPE>
bool SimpleH5File::datasetDataType( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  if( !isDataset(path) )
    return false;

  std::string dataset_path( simplifyPath(path) );

  HId dataset( H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT) );
  if( dataset.id < 0 )
  {
    return false;
  }
  HId datatype( H5Tcopy(dataset.id) );
  HId native_type( nativeToH5Type<NATIVE_TYPE>::type() );
  htri_t tri = H5Tequal( datatype.id, native_type.id);
  bool type_compatible = tri > 0;
  return type_compatible;
}
template bool SimpleH5File::datasetDataType<char>( const std::string& );
template bool SimpleH5File::datasetDataType<unsigned char>( const std::string& );
template bool SimpleH5File::datasetDataType<short>( const std::string& );
template bool SimpleH5File::datasetDataType<unsigned short>( const std::string& );
template bool SimpleH5File::datasetDataType<int>( const std::string& );
template bool SimpleH5File::datasetDataType<unsigned int>( const std::string& );
template bool SimpleH5File::datasetDataType<long>( const std::string& );
template bool SimpleH5File::datasetDataType<unsigned long>( const std::string& );
template bool SimpleH5File::datasetDataType<float>( const std::string& );
template bool SimpleH5File::datasetDataType<double>( const std::string& );


bool SimpleH5File::exists( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string tmp = simplifyPath(path);
  std::vector<std::string> path_vec( pathToVector(tmp) );
  tmp = std::string();

  for( int i = 0; i < (int)path_vec.size(); ++i )
  {
    tmp += "/" + path_vec[i];
    htri_t status = H5Lexists(file_id, tmp.c_str(), H5P_DEFAULT );
    if( status <= 0 )
      return false;
  }
  return true;
}




namespace{
  herr_t add_objects_to_vector( hid_t g_id, const char* name,
                                const H5L_info_t* info, void* op_data)
  {
    H5O_info_t obj_info;
    herr_t status = H5Oget_info_by_name(g_id, name, &obj_info, H5P_DEFAULT);
    if( status >= 0 )
    {
      // consider only groups and datasets as objects
      if( obj_info.type == H5O_TYPE_DATASET || obj_info.type == H5O_TYPE_GROUP )
      {
        std::vector<std::string>* result = reinterpret_cast<std::vector<std::string>*>(op_data);
        result->push_back(std::string(name));
      }
    }
    return 0;
  }

  herr_t add_groups_to_vector( hid_t g_id, const char* name,
                               const H5L_info_t* info, void* op_data)
  {
    H5O_info_t obj_info;
    herr_t status = H5Oget_info_by_name(g_id, name, &obj_info, H5P_DEFAULT);
    if( status >= 0 )
    {
      // consider only groups and datasets as objects
      if( obj_info.type == H5O_TYPE_GROUP )
      {
        std::vector<std::string>* result = reinterpret_cast<std::vector<std::string>*>(op_data);
        result->push_back(std::string(name));
      }
    }
    return 0;
  }

  herr_t add_datasets_to_vector( hid_t g_id, const char* name,
                                 const H5L_info_t* info, void* op_data)
  {
    H5O_info_t obj_info;
    herr_t status = H5Oget_info_by_name(g_id, name, &obj_info, H5P_DEFAULT);
    if( status >= 0 )
    {
      // consider only groups and datasets as objects
      if( obj_info.type == H5O_TYPE_DATASET )
      {
        std::vector<std::string>* result = reinterpret_cast<std::vector<std::string>*>(op_data);
        result->push_back(std::string(name));
      }
    }
    return 0;
  }

  herr_t add_attribute_to_vector( hid_t loc_id, const char* attr_name, 
                                  const H5A_info_t* info, void* op_data)
  {
    std::vector<std::string>* result = reinterpret_cast<std::vector<std::string>*>(op_data);
    result->push_back(std::string(attr_name));
    return 0;
  }

}


std::vector<std::string> SimpleH5File::listObjects( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::vector<std::string> result;

  std::string tmp = simplifyPath(path);
  hsize_t idx = 0;
  herr_t status = H5Literate_by_name( file_id, tmp.c_str(), 
                                      H5_INDEX_NAME, H5_ITER_NATIVE, 
                                      &idx, 
                                      add_objects_to_vector, 
                                      (void*)&result, H5P_DEFAULT );
  if( status != 0 )
  {
    // throw?
  }
  return result;
}


std::vector<std::string> SimpleH5File::listDatasets( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::vector<std::string> result;

  std::string tmp = simplifyPath(path);
  hsize_t idx = 0;
  herr_t status = H5Literate_by_name( file_id, tmp.c_str(), 
                                      H5_INDEX_NAME, H5_ITER_NATIVE, 
                                      &idx, 
                                      add_datasets_to_vector, 
                                      (void*)&result, H5P_DEFAULT );
  if( status != 0 )
  {
    // throw?
  }
  return result;
}


std::vector<std::string> SimpleH5File::listGroups( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::vector<std::string> result;

  std::string tmp = simplifyPath(path);
  hsize_t idx = 0;
  herr_t status = H5Literate_by_name( file_id, tmp.c_str(), 
                                      H5_INDEX_NAME, H5_ITER_NATIVE, 
                                      &idx, 
                                      add_groups_to_vector, 
                                      (void*)&result, H5P_DEFAULT );
  if( status != 0 )
  {
    // throw?
  }
  return result;
}


std::vector<std::string> SimpleH5File::listAttributes( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::vector<std::string> result;

  std::string tmp = simplifyPath(path);
  hsize_t idx = 0;
  herr_t status = H5Aiterate_by_name( file_id, tmp.c_str(), 
                                      H5_INDEX_NAME, H5_ITER_NATIVE, 
                                      &idx, 
                                      add_attribute_to_vector, 
                                      (void*)&result, H5P_DEFAULT );
  if( status != 0 )
  {
    // throw?
  }
  return result;
}






template <class T>
void SimpleH5File::writeDataset( const T* data, 
                                 const std::vector<size_t>& dims, 
                                 const std::string& path,
                                 Compression compress )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE;

  std::string dataset_path( simplifyPath(path) );

  if( exists(dataset_path) && isDataset(dataset_path) )
  {
    // check if the existing dataset is compatible in extents and type
    HId dataset( H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT) );
    HId datatype( H5Tcopy(dataset.id) );
    HId native_type ( nativeToH5Type<T>::type() );
    htri_t tri = H5Tequal( datatype.id, native_type.id);
    bool type_compatible = tri > 0;
    if( type_compatible && dims == getDatasetExtents(dataset_path) )
    {
      herr_t status = H5Dwrite(dataset.id, datatype.id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                               (void*) data);
      if( status < 0 )
      {
        throw std::runtime_error("H5Dwrite failed");
      }
      return;
    }
    else
    {
      remove(dataset_path);
    }
  }

  // create a new dataset if necessary
  if( !exists(dataset_path) )
  {
    createDataset<T>(dataset_path, dims, compress);
    HId dataset( H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT) );
    if( dataset.id < 0 )
    {
      throw std::runtime_error("Cannote open dataset");
    }

    HId datatype( nativeToH5Type<T>::type() );
    herr_t status = H5Dwrite(dataset.id, datatype.id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      (void*) data);
    if( status < 0 )
    {
      throw std::runtime_error("H5Dwrite failed");
    }
  }

}
template void SimpleH5File::writeDataset( const char* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const unsigned char* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const short* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const unsigned short* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const int* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const unsigned int* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const long* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const unsigned long* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const float* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );
template void SimpleH5File::writeDataset( const double* , 
                                          const std::vector<size_t>&, 
                                          const std::string&,
                                          Compression );



template <class T>
void SimpleH5File::readDataset( T* data, const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN

  std::string dataset_path( simplifyPath(path) );
  HId dataset( H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT) );
  if( dataset.id < 0 )
  {
    throw std::runtime_error("Cannot open dataset");
  }
  HId datatype( H5Tcopy(dataset.id) );
  HId native_type( nativeToH5Type<T>::type() );
  htri_t tri = H5Tequal( datatype.id, native_type.id);
  bool type_compatible = tri > 0;
  if( !type_compatible )
  {
    throw std::runtime_error("Cannot read from dataset, types are not compatible");
  }

  herr_t status = H5Dread(dataset.id, native_type.id, 
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, (void*) data);
  if( status < 0 )
  {
    throw std::runtime_error("Read from dataset failed");
  }
}
template void SimpleH5File::readDataset( char*, const std::string& );
template void SimpleH5File::readDataset( unsigned char*, const std::string& );
template void SimpleH5File::readDataset( short*, const std::string& );
template void SimpleH5File::readDataset( unsigned short*, const std::string& );
template void SimpleH5File::readDataset( int*, const std::string& );
template void SimpleH5File::readDataset( unsigned int*, const std::string& );
template void SimpleH5File::readDataset( long*, const std::string& );
template void SimpleH5File::readDataset( unsigned long*, const std::string& );
template void SimpleH5File::readDataset( float*, const std::string& );
template void SimpleH5File::readDataset( double*, const std::string& );



template <class T>
void SimpleH5File::writeAttribute( const T& value, 
                                   const std::string& attr_name, 
                                   const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE

  std::vector<T> tmp(1, value);
  writeAttribute(tmp, attr_name, path);
}
template void SimpleH5File::writeAttribute( const char& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const unsigned char& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const short& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const unsigned short& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const int& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const unsigned int& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const long& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const unsigned long& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const float& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::writeAttribute( const double& , 
                                  const std::string&, const std::string& path );



template <class T>
void SimpleH5File::writeAttribute( const std::vector<T>& value, 
                                   const std::string& attr_name, 
                                   const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE;

  if( existsAttribute(attr_name, path) )
  {
    removeAttribute(attr_name, path);
  }

  std::string obj_path( simplifyPath(path) );
  HId object( H5Oopen(file_id, obj_path.c_str(), H5P_DEFAULT) );

  HId datatype( nativeToH5Type<T>::type() );
  hsize_t dims = value.size();
  HId dataspace( H5Screate_simple(1,&dims,0) );
  HId attr( H5Acreate2(object.id, attr_name.c_str(), datatype.id, dataspace.id, 
                       H5P_DEFAULT, H5P_DEFAULT) );

  const void* buf = &(value[0]);
  H5Awrite( attr.id, datatype.id, buf );
}



void SimpleH5File::writeAttribute( const char str[],
                                   const std::string& attr_name, 
                                   const std::string& path )
{
  writeAttribute(std::string(str), attr_name, path);
}



void SimpleH5File::writeAttribute( const std::string& str,
                                   const std::string& attr_name, 
                                   const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE;

  if( existsAttribute(attr_name, path) )
  {
    removeAttribute(attr_name, path);
  }

  HId datatype( H5Tcopy(H5T_C_S1) );
  herr_t status = H5Tset_size(datatype.id, str.length()+1);
  if( status < 0 )
  {
    throw std::runtime_error("H6Tset_size failed");
  }

  HId dataspace( H5Screate( H5S_SCALAR ) );

  std::string obj_path( simplifyPath(path) );
  HId object( H5Oopen(file_id, obj_path.c_str(), H5P_DEFAULT) );

  HId attr( H5Acreate2(object.id, attr_name.c_str(), 
                            datatype.id, dataspace.id, 
                            H5P_DEFAULT, H5P_DEFAULT ) );

  status = H5Awrite(attr.id, datatype.id, (const void*) str.c_str());

}



template <class T>
void SimpleH5File::readAttribute( T& value, 
                                  const std::string& attr_name, 
                                  const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN

  std::vector<T> tmp;
  readAttribute(tmp, attr_name, path);
  if( tmp.size() != 1 )
  {
    throw std::runtime_error("Cannot read attribute");
  }
  value = tmp[0];
}
template void SimpleH5File::readAttribute( char& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( unsigned char& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( short& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( unsigned short& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( int& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( unsigned int& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( long& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( unsigned long& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( float& , 
                                  const std::string&, const std::string& path );
template void SimpleH5File::readAttribute( double& , 
                                  const std::string&, const std::string& path );



template <class T>
void SimpleH5File::readAttribute( std::vector<T>& value, 
                                  const std::string& attr_name, 
                                  const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string obj_path( simplifyPath(path) );
  HId object( H5Oopen(file_id, obj_path.c_str(), H5P_DEFAULT) );
  if( object.id < 0 )
  {
    throw std::runtime_error("Cannot open object");
  }
  HId attr( H5Aopen(object.id, attr_name.c_str(), H5P_DEFAULT) );
  if( attr.id < 0 )
  {
    throw std::runtime_error("Cannot open object");
  }

  HId datatype( H5Aget_type(attr.id) );
  HId native_type( nativeToH5Type<T>::type() );
  htri_t tri = H5Tequal(datatype.id, native_type.id);
  if( tri > 0 )
  {
    HId dataspace( H5Aget_space(attr.id) );
    int ndims = H5Sget_simple_extent_ndims(dataspace.id);
    if( ndims != 1 )
    {
      throw std::runtime_error("Cannot read attributes with dimension > 1");
    }
    hsize_t dims;

    H5Sget_simple_extent_dims(dataspace.id,&dims,0);

    T* buf = new T[dims];
    H5Aread(attr.id, native_type.id, (void*) buf);

    value.resize(dims);
    for( int i = 0; i < (int)dims; ++i )
      value[i] = buf[i];

    delete[] buf;
  }
}


void SimpleH5File::readAttribute( std::string& str, 
                                  const std::string& attr_name, 
                                  const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string obj_path( simplifyPath(path) );
  HId object( H5Oopen(file_id, obj_path.c_str(), H5P_DEFAULT) );
  if( object.id < 0 )
  {
    throw std::runtime_error("Cannot open object");
  }
  HId attr( H5Aopen(object.id, attr_name.c_str(), H5P_DEFAULT) );
  if( attr.id < 0 )
  {
    throw std::runtime_error("Cannot open attribue");
  }

  HId datatype( H5Aget_type(attr.id) );
  if( H5T_STRING == H5Tget_class(datatype.id) )
  {
    HId dataspace( H5Aget_space(attr.id) );
    if( H5S_SCALAR != H5Sget_simple_extent_type(dataspace.id) )
    {
      throw std::runtime_error("Attribute is not scalar");
    }
    if( H5Tis_variable_str(datatype.id) > 0 )
    {
      hsize_t dims = 0;
      int ndims = H5Sget_simple_extent_dims(dataspace.id,&dims,0);
      if( ndims == 0 )
        dims = 1;
      typedef char* char_p;
      char_p* rdata = new char_p[dims];
      HId memtype( H5Tcopy(H5T_C_S1) );
      herr_t status = H5Tset_size(memtype.id, H5T_VARIABLE);

      status = H5Aread(attr.id, memtype.id, rdata);
      if( status < 0 )
      {
        throw std::runtime_error("H5Aread failed");
      }
      str = std::string( rdata[0] );
      H5Dvlen_reclaim (memtype.id, dataspace.id, H5P_DEFAULT, rdata);
      delete[] rdata;
    }
    else // fixed length string
    {
      size_t size = H5Tget_size(datatype.id);
      char* buf = new char[size];
      herr_t status = H5Aread(attr.id, datatype.id, (void*)buf);
      if( status < 0 )
      {
        throw std::runtime_error("H5Aread failed");
      }

      H5T_str_t str_pad = H5Tget_strpad(datatype.id);
      if( H5T_STR_NULLTERM == str_pad )
      {
        str = std::string( buf );
      }
      else if( H5T_STR_NULLPAD == str_pad )
      {
        str = std::string( buf, size );
      }
      delete[] buf;
    }
  }

}


template <class T>
void SimpleH5File::createDataset( const std::string& dataset_path, 
                                  const std::vector<size_t>& dims, 
                                  Compression compress )
{
  LOCK_GUARD
  if( dims.size() > SIMPLEH5FILE_MAXIMUM_NUMBER_OF_DIMENSIONS )
  {
    throw std::runtime_error("Maximum number of dimensions exceeded");
  }

  hid_t lcpl_id = H5Pcreate( H5P_LINK_CREATE );
  if( lcpl_id == -1 )
  {
    throw std::runtime_error("Creating link creation property list failed");
  }
  herr_t status = H5Pset_create_intermediate_group( lcpl_id, 1 );
  if( status < 0 )
  {
    H5Pclose(lcpl_id);
    throw std::runtime_error("H5Pset_create_intermediate_group failed");
  }

  hsize_t current_dims[SIMPLEH5FILE_MAXIMUM_NUMBER_OF_DIMENSIONS];
  for( size_t i = 0; i < dims.size(); ++i )
  {
    current_dims[i] = dims[i];
  }

  HId dataspace( H5Screate_simple(dims.size(), current_dims, 0) );
  if( dataspace.id < 0 )
  {
    throw std::runtime_error("Cannot create dataspace");
  }

  HId datatype( nativeToH5Type<T>::type() );
  HId dataset( H5Dcreate2( file_id, dataset_path.c_str(), 
                                 datatype.id, dataspace.id, 
                                 lcpl_id, H5P_DEFAULT, H5P_DEFAULT ) );
  H5Pclose(lcpl_id);
  if( dataset.id < 0 )
  {
    throw std::runtime_error("Cannot create dataset");
  }

}




std::pair<size_t,size_t> SimpleH5File::getDatasetOffsetAndSize( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  HId dataset( H5Dopen2(file_id, path.c_str(), H5P_DEFAULT) );
  if( dataset.id < 0 )
  {
    throw std::runtime_error("Cannot open dataset");
  }
  haddr_t status = H5Dget_offset(dataset.id);
  if( status < 0 )
  {
    throw std::runtime_error("Cannot get dataset size");
  }

  size_t count = 1;
  std::vector<size_t> extents = getDatasetExtents(path);
  for( size_t e : extents )
    count *= e;

  return std::make_pair(size_t(status),count);
}


bool SimpleH5File::isDatasetContiguous( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  HId dataset( H5Dopen2(file_id, path.c_str(), H5P_DEFAULT) );
  if( dataset.id < 0 )
  {
    throw std::runtime_error("Cannot open dataset");
  }

  hid_t dcpl = H5Dget_create_plist(dataset.id);
  if( dcpl < 0 )
  {
    throw std::runtime_error("Cannot get access property list");
  }
  H5D_layout_t layout = H5Pget_layout(dcpl);
  if( layout < 0 )
  {
    H5Pclose(dcpl);
    throw std::runtime_error("Cannot get dataset layout");
  } 
  H5Pclose(dcpl);

  return layout == H5D_CONTIGUOUS;
}


std::vector<size_t> SimpleH5File::getDatasetExtents( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  hid_t dataset_id = H5Dopen2(file_id, path.c_str(), H5P_DEFAULT);
  if( dataset_id < 0 )
  {
    throw std::runtime_error("Cannot open dataset");
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  if( dataspace_id < 0 )
  {
    H5Dclose(dataset_id);
    throw std::runtime_error("Cannot get dataspace");
  }
  H5Sselect_all(dataspace_id);

  int ndims = H5Sget_simple_extent_ndims(dataspace_id);
  if( ndims < 0 )
  {
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    throw std::runtime_error("Cannot get extents");
  }
  hsize_t* dims = new hsize_t[ndims];
  H5Sget_simple_extent_dims(dataspace_id, dims, 0);

  std::vector<size_t> result(ndims);
  for( int i = 0; i < ndims; ++i )
    result[i] = dims[i];

  delete[] dims;

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);

  return result;
}






void SimpleH5File::removeAttribute( const std::string& attr_name, 
                                    const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN_AND_WRITE

  herr_t status = H5Adelete_by_name( file_id,
                                     simplifyPath(path).c_str(), 
                                     attr_name.c_str(),
                                     H5P_DEFAULT );
  if( status < 0 )
  {
    throw std::runtime_error("Deleting attribute failed");
  }
}



bool SimpleH5File::existsAttribute( const std::string& attr_name,
                                    const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  htri_t tri = H5Aexists_by_name( file_id, 
                                  simplifyPath(path).c_str(),
                                  attr_name.c_str(),
                                  H5P_DEFAULT );
  return tri > 0;
}


H5O_info_t SimpleH5File::getObjectInfo( const std::string& path )
{
  LOCK_GUARD
  SIMPLEH5FILE_CHECK_OPEN;

  std::string obj_path( simplifyPath(path) );

  if( !exists(obj_path) )
  {
    throw std::runtime_error("SimpleH5File::getObjectInfo(): Object does not exist!");
  }

  H5O_info_t result = H5O_info_t();
  int status = H5Oget_info_by_name(file_id, obj_path.c_str(), &result, H5P_DEFAULT);
  if( status < 0 )
    throw std::runtime_error("H5Oget_info_by_name() failed");

  return result;
}


bool SimpleH5File::isHDF5( const std::string& filename )
{
  LockGuard<std::recursive_mutex> lock(global_SimpleH5File_mutex,true);
  htri_t status = H5Fis_hdf5(filename.c_str());
  if( status > 0 )
    return true;
  else
    return false;
}


std::string SimpleH5File::simplifyPath( const std::string& path )
{
  std::string result( path );

  // remove multiple '/'
  size_t pos = result.find("//");
  while( pos != std::string::npos )
  {
    result.erase(pos,1);
    pos = result.find("//");
  }

  // remove leading and trailing whitespace
  pos = result.find_first_not_of(' ');
  if( pos > 0 && pos != std::string::npos )
    result.erase(0,pos);

  pos = result.find_last_not_of(' ');
  if( pos < result.size()-1 && pos != std::string::npos )
    result.erase(pos+1);

  //std::cout << result << std::endl;
  return result;
}





                    


