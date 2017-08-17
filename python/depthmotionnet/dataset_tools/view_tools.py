#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import pyximport; pyximport.install()
import numpy as np

from .view import View

def compute_visible_points_mask( view1, view2, borderx=0, bordery=0 ):
    """Computes a mask of the pixels in view1 that are visible in view2

    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    borderx: int
        border in x direction. Points in the border are considered invalid

    bordery: int
        border in y direction. Points in the border are considered invalid

    Returns a mask of valid points
    """
    from .view_tools_cython import compute_visible_points_mask as _compute_visible_points_mask
    assert view1.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"
    return _compute_visible_points_mask( view1, view2, borderx, bordery )


def compute_depth_ratios( view1, view2 ):
    """Projects each point defined in view1 to view2 and computes the ratio of 
    the depth value of the projected point and the stored depth value in view2.


    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    Returns the scale value for view2 relative to view1
    """
    from .view_tools_cython import compute_depth_ratios as _compute_depth_ratios
    return _compute_depth_ratios(view1, view2)


def check_depth_consistency( view, rest_of_the_views, depth_ratio_threshold=0.9, min_valid_threshold=0.5, min_depth_consistent=0.7 ):
    """Checks if the depth of view is consistent with the rest_of_the_views
    
    view: View namedtuple
        Reference view

    rest_of_the_views: list of View namedtuple
        List of the rest of the views

    depth_ratio_threshold: float
        The allowed minimum depth ratio

    min_valid_threshold: float
        ratio of pixels that should have consistent depth values with the rest_of_the_views

    min_depth_consistent: float
        ratio of depth consistent pixels with respect to the number of valid depth ratios

    Returns True if the depth is consistent
    """
    min_ratio_threshold = min(depth_ratio_threshold, 1/depth_ratio_threshold)
    max_ratio_threshold = max(depth_ratio_threshold, 1/depth_ratio_threshold)
    for v in rest_of_the_views:
        dr = compute_depth_ratios(view, v)
        valid_dr = dr[np.isfinite(dr)]
        if valid_dr.size / dr.size < min_valid_threshold:
            return False

        num_consistent = np.count_nonzero((valid_dr > min_ratio_threshold) & (valid_dr < max_ratio_threshold))
        if num_consistent / valid_dr.size < min_depth_consistent:
            return False

    return True


def create_image_overview( views ):
    """Creates a small overview image showing the RGB images of all views
    
    views: list of View  or  list of list of View

    Returns a PIL.Image
    """
    assert isinstance(views, list)
    from .helpers import concat_images_vertical, concat_images_horizontal
    max_height = 100 # maximum height of individual images

    def resize_image(img):
        if img.size[1] > max_height:
            new_width = int(img.size[0]*(max_height/img.size[1]))
            return img.resize((new_width,max_height))
        else:
            return img

    column_images = []
    for col in views:
        if isinstance(col,list):
            tmp_images = []
            for row in col:
                tmp_images.append(resize_image(row.image))
            col_img = concat_images_vertical(tmp_images)
            column_images.append(col_img)
        elif isinstance(col,View):
            column_images.append(resize_image(col.image))
    return concat_images_horizontal(column_images)


def visualize_views( views ):
    """Visualizes views

    views: list of View namedtuple

    Opens a vtk window with the visualization
    """
    import vtk
    from .. import vis


    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)

    axes = vtk.vtkAxesActor()
    axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    axes.SetCylinderRadius(0.03)
    axes.SetShaftTypeToCylinder()
    renderer.AddActor(axes)

    renwin = vtk.vtkRenderWindow()
    renwin.SetWindowName("Viewer (press 'm' to change colors, use '.' and ',' to adjust opacity)")
    renwin.SetSize(800,600)
    renwin.AddRenderer(renderer)
    
 
    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interstyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interstyle)
    interactor.SetRenderWindow(renwin)

    colors = ((1,0,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0), (1,1,1), (0,1,0))
 
    pointcloud_polydatas = []
    pointcloud_actors = []
    for idx, view in enumerate(views):

        img_arr = None
        if not view.image is None:
            img_arr = np.array(view.image).transpose([2,0,1])


        pointcloud = vis.compute_point_cloud_from_depthmap(view.depth, view.K, view.R, view.t, colors=img_arr)
        pointcloud_polydata = vis.create_pointcloud_polydata( 
            points=pointcloud['points'], 
            colors=pointcloud['colors'] if 'colors' in pointcloud else None,
        )
        pointcloud_polydatas.append(pointcloud_polydata)

        pc_mapper = vtk.vtkPolyDataMapper()
        pc_mapper.SetInputData(pointcloud_polydata)

        pc_actor = vtk.vtkActor()
        pointcloud_actors.append(pc_actor)
        pc_actor.SetMapper(pc_mapper)
        pc_actor.GetProperty().SetPointSize(2)
        

        color = colors[idx%len(colors)]

        pc_actor.GetProperty().SetColor(*color)
        renderer.AddActor(pc_actor)

        cam_actor = vis.create_camera_actor(view.R,view.t)
        cam_actor.GetProperty().SetColor(*color)
        renderer.AddActor(cam_actor)



    def change_point_properties(obj, ev):
        if change_point_properties.current_active_scalars == "Colors":
            change_point_properties.current_active_scalars = ""
        else:
            change_point_properties.current_active_scalars = "Colors"

        if "m" == obj.GetKeySym():
            for polydata in pointcloud_polydatas:
                polydata.GetPointData().SetActiveScalars(change_point_properties.current_active_scalars)

        if "period" == obj.GetKeySym():
            for actor in pointcloud_actors:
                opacity = actor.GetProperty().GetOpacity()
                opacity = min(1.0, opacity - 0.1)
                
                actor.GetProperty().SetOpacity(opacity)
        if "comma" == obj.GetKeySym():
            for actor in pointcloud_actors:
                opacity = actor.GetProperty().GetOpacity()
                opacity = max(0.0, opacity + 0.1)
                actor.GetProperty().SetOpacity(opacity)
        renwin.Render()
            
    change_point_properties.current_active_scalars = "Colors"

    interactor.AddObserver('KeyReleaseEvent', change_point_properties)
    
    # Start
    interactor.Initialize()
    interactor.Start()

    interactor.RemoveAllObservers()
    del change_point_properties
    

