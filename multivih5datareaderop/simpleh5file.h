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
#ifndef SIMPLEH5FILE_H_
#define SIMPLEH5FILE_H_
#include <vector>
#include <string>
#include <hdf5.h>


/*!
 *  This class provides basic functions to manipulate hdf5 files
 */
class SimpleH5File
{
public:
  //! File modes
  enum FileMode { TRUNCATE,   //!< Create or overwrite file
                  READ,       //!< Read only file access
                  READ_WRITE  //!< Read write access
                };
  //! Compression settings
  enum Compression { UNCOMPRESSED, /*GZIP_1, GZIP_2,...*/ };
  
  /*!
   *  ctor.
   *  \param use_locking  If true uses mutexes to allow accessing the object
   *                      from multiple threads.
   */
  SimpleH5File(bool use_locking=false);


  /*!
   *  Creates a SimpleH5File object and opens the specified hdf5 file.
   *  
   *  \param filename Filename of the hdf5 file. E.g. 'myfile.h5'
   *  \param mode     The mode for opening the file.
   *  \param use_locking  If true uses mutexes to allow accessing the object
   *                      from multiple threads.
   */
  SimpleH5File( const std::string& filename, FileMode mode=READ, bool use_locking=false );


  /*!
   *  dtor. Closes the opened file.
   */
  virtual ~SimpleH5File();


  /*!
   *  Opens the specified hdf5 file. If a hdf5 file was already opened it is 
   *  closed before opening the new file.
   *  
   *  \param filename Filename of the hdf5 file. E.g. 'myfile.h5'
   *  \param mode     The mode for opening the file.
   */
  void open( const std::string& filename, FileMode mode=READ );


  /*!
   *  Returns whether a file is open.
   */
  bool isOpen() const;


  /*!
   *  Returns whether the object uses locking to allow multiple thread using
   *  this object simultaneously.
   */
  bool useLocking() const;


  /*!
   *  Closes the file. Has no effect if no file is open.
   */
  void close();


  /*!
   *  Creates a new group and creates parent directories if necessary.
   *
   *  \param path  Path of the new group. E.g. '/group1/group2/newGroup'
   *               creates 'newGroup' and 'group1', 'group2' if they dont exist.
   */
  void makeGroup( const std::string& path );


  /*!
   *  Removes a group or dataset
   *
   *  \param path  The path of the group or dataset to be removed.
   *               E.g. '/group/mydataset' removes 'mydataset'
   */
  void remove( const std::string& path );


  /*!
   *  Returns whether the object with the specified path is a group.
   */
  bool isGroup( const std::string& path );


  /*!
   *  Returns whether the object with the specified path is a dataset.
   */
  bool isDataset( const std::string& path );

  /*!
   *  Returns true if the native type and the dataset type match.
   *  If 'path' is not a dataset then false is returned.
   */
  template <class NATIVE_TYPE>
  bool datasetDataType( const std::string& path );

  /*!
   *  Returns whether the path points to a dataset or group
   */
  bool exists( const std::string& path );


  /*!
   *  Lists all objects (datasets and groups) with the parent specified by path.
   *
   *  \param path  E.g. '/' lists all groups and datasets of the root group.
   */
  std::vector<std::string> listObjects( const std::string& path );


  /*!
   *  Lists all datasets with the parent specified by path.
   *
   *  \param path  E.g. '/' lists all datasets of the root group.
   */
  std::vector<std::string> listDatasets( const std::string& path );


  /*!
   *  Lists all groups with the parent specified by path.
   *
   *  \param path  E.g. '/' lists all groups of the root group.
   */
  std::vector<std::string> listGroups( const std::string& path );


  /*!
   *  Lists all attributes of a dataset or group specified by path.
   *
   *  \param path  E.g. '/mydataset' lists all attributes of 'mydataset'.
   */
  std::vector<std::string> listAttributes( const std::string& path );



  /*!
   *  Writes a dataset. Any existing dataset will be overwritten.
   *  This command will also create parent groups if necessary.
   *
   *  \param data      Pointer to the data
   *  \param dims      Dimensions of the dataset to write. The extent of each 
   *                   dimension is defined in elements.
   *  \param path      Path to the dataset e.g. '/group/dataset'.
   *  \param compress  Reserved for future use to specify the compression filter
   */
  template <class T>
  void writeDataset( const T* data, const std::vector<size_t>& dims, 
                     const std::string& path,
                     Compression compress = UNCOMPRESSED );

  /*!
   *  Reads the dataset to the given buffer.
   *
   *  \param data  The buffer for reading the dataset. The buffer must be 
   *               allocated by the user. Use getDatasetExtents() to retrieve
   *               the size of the dataset.
   *  \param path  Path to the dataset e.g. '/group/dataset'.
   */
  template <class T>
  void readDataset( T* data, const std::string& path );


  /*!
   *  Returns the byte offset of the dataset in the file and the number 
   *  of elements.
   *
   *  \param path  Path to the dataset e.g. '/group/dataset'.
   *  \return Returns the byte offset of the dataset in the file and the number 
   *  of elements
   */
  std::pair<size_t,size_t> getDatasetOffsetAndSize( const std::string& path );


  /*!
   *  Returns whether the dataset is contiguous or not.
   *
   *  \param path  Path to the dataset e.g. '/group/dataset'.
   *  \return Returns true if the dataset is contiguous
   */
  bool isDatasetContiguous( const std::string& path );


  /*!
   *  Returns the extents of the dataset.
   *
   *  \param path  Path to the dataset e.g. '/group/dataset'.
   *  \return Returns a vector containing the extents. The size of the vector
   *          corresponds to the number of dimensions of the dataset
   */
  std::vector<size_t> getDatasetExtents( const std::string& path );


  /*!
   *  Writes an attribute. An attribute is attached to a group or a dataset.
   *  Overwrites existing attributes.
   *
   *  \param value      The value of the attribute.
   *  \param attr_name  The name of the attribute e.g. 'my_int_attribute'
   *  \param path       The path of the group or dataset e.g. '/group'
   */
  template <class T>
  void writeAttribute( const T& value, 
                       const std::string& attr_name, const std::string& path ); 


  //! \sa writeAttribute(const T& value, const std::string&, const std::string&)
  template <class T>
  void writeAttribute( const std::vector<T>& value, 
                       const std::string& attr_name, const std::string& path ); 

  //! \sa writeAttribute(const T& value, const std::string&, const std::string&)
  void writeAttribute( const char str[],
                       const std::string& attr_name, const std::string& path ); 

  //! \sa writeAttribute(const T& value, const std::string&, const std::string&)
  void writeAttribute( const std::string& str,
                       const std::string& attr_name, const std::string& path ); 


  /*!
   *  Reads an attribute. An attribute is attached to a group or a dataset.
   *
   *  \param value      The value that is written to the attribute
   *  \param attr_name  The name of the attribute e.g. 'my_int_attribute'
   *  \param path       The path of the group or dataset e.g. '/group'
   */
  template <class T>
  void readAttribute( T& value, 
                      const std::string& attr_name, const std::string& path ); 

  //! \sa readAttribute(T& value, const std::string&, const std::string&)
  template <class T>
  void readAttribute( std::vector<T>& value, 
                      const std::string& attr_name, const std::string& path ); 

  //! \sa readAttribute(T& value, const std::string&, const std::string&)
  void readAttribute( std::string& str,
                      const std::string& attr_name, const std::string& path ); 

  /*!
   *  Removes an attribute.
   *
   *  \param attr_name  Name of the attribute.
   *  \param path       The path of the group or dataset e.g. '/group'
   */
  void removeAttribute( const std::string& attr_name, const std::string& path );
  

  /*!
   *  Checks the existence of an attribute.
   *
   *  \param attr_name  Name of the attribute.
   *  \param path       The path of the group or dataset e.g. '/group'
   *  \return Returns true if the attribute exists.
   */
  bool existsAttribute( const std::string& attr_name, const std::string& path );


  /*!
   *  Returns the H5O_info_t struct for the object with the specified path.
   *
   *  \param path  Path to an object (group or dataset) e.g. '/mydataset'
   *  \return The H5O_info_t struct of the object.
   */
  H5O_info_t getObjectInfo( const std::string& path );


  /*!
   *  Checks if a file is a hdf5 file
   *
   *  \param filename path to the file
   *  \return Returns true if the file is a hdf5 file.
   *          Returns false if the file is not a hdf5 file.
   *          Returns false if the file does not exist or reading fails.
   */
  static bool isHDF5( const std::string& filename );


  /*!
   *  Simplifies a hdf5 path. This function removes leading and trailing 
   *  whitespaces and removes rendundant multiple '/'.
   *
   *  \return The simplified path.
   */
  static std::string simplifyPath( const std::string& path );

protected:

  FileMode mode;

  hid_t file_id; //! hdf5 file identifier

private:
  SimpleH5File( const SimpleH5File& other ):use_locking(false) {}
  SimpleH5File& operator=( const SimpleH5File& other ) { return *this; }


  /*!
   *  Creates a dataset. This command will also create parent groups if 
   *  necessary.
   *
   *  \param dataset_path  Path to the dataset e.g. '/group/dataset'.
   *  \param dims          Dimensions of the dataset to write. The extent of 
   *                       each dimension is defined in elements.
   *  \param compress      Reserved for future use to specify the compression
   *                       filter
   */
  template <class T>
  void createDataset( const std::string& dataset_path, 
                      const std::vector<size_t>& dims, 
                      Compression compress = UNCOMPRESSED );

  bool is_open;
  const bool use_locking;


};





#endif /* SIMPLEH5FILE_H_ */
