import numpy as np
from model import *
from bamg import *
from transient import *
from parameterize import *
from asymetric_geometries2 import Geometry
from solve import *
from InterpFromMeshToMesh2d import *
from setflowequation import *
from setmask import *
from SetMarineIceSheetBC import *
from generate_exp2 import *
from os import path
from cuffey import *
from transientrestart2 import *
from loadmodel import *
from calvingvonmises import *


def SpinUp(params, run_name, load_name):
    y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()
    start_icefront=params['start_icefront']
    slab_thickness=params['slab_thickness']
    steepness=params['steepness']
    null_level=params['null_level']
    x_dim=params['x_dim']


    generateEXP(params, x_dim, y_dim) 
    md=bamg(model(), 'domain','./exp_file2.exp', 'hmax',100, 'hmin', 100) 
    #md=bamg(model(), 'domain', './Square2.exp', 'hmax', 100, 'hmin', 100) 
    md.geometry.bed=Geometry(md, y_dim+20000, x_dim, slope, params['bump_spread'], params['bump_height'], params['bump_pos'], params['bump_skew'],steepness, gap_halfwidth, dc, params['bay_spread1'], params['bay_height1'], params['bay_pos1'],params['bay_skew1'], params['bay_spread2'], params['bay_height2'], params['bay_pos2'],params['bay_skew2'], step)
    old_mesh_elements=md.mesh.elements
    old_mesh_x=md.mesh.x
    old_mesh_y=md.mesh.y
    old_mesh_geometry=md.geometry.bed
    
    h=np.nan*np.ones(md.mesh.numberofvertices)
    h[np.where(np.logical_and(np.logical_and(md.mesh.y<17800, md.mesh.y>12200), np.logical_and(md.mesh.x<62000, md.mesh.x>40000)))]=100
    #h[np.where(np.logical_and(md.geometry.bed<1200, np.logical_and(md.mesh.x<50000, md.mesh.x>18000)))]=100

    md=bamg(md, 'field', old_mesh_geometry, 'hmax', 2000, 'hmin', params['hmin'], 'gradation', 1.5, 'hVertices', h)
    md.miscellaneous.name=run_name

    md.geometry.bed=InterpFromMeshToMesh2d(old_mesh_elements, old_mesh_x, old_mesh_y, old_mesh_geometry, md.mesh.x, md.mesh.y)[0][:,0]+null_level  
    md.timestepping.final_time=params['final_time']
    md.timestepping.time_step=params['timestepping']
    md.settings.output_frequency=params['output_frequency']

    md.geometry.surface=md.geometry.bed+1
    surface_pos=np.where(md.mesh.x<=start_icefront)
    #md.geometry.surface[surface_pos]=slab_thickness+null_level+md.mesh.x[surface_pos]*slope
    md.geometry.surface[surface_pos]=np.sqrt(2*(start_icefront-md.mesh.x[surface_pos]))-null_level+150+md.mesh.x[surface_pos]*slope

    below_pos=np.where(md.geometry.surface<md.geometry.bed+1)
    md.geometry.surface[below_pos]=md.geometry.bed[below_pos]+1
    
    md.geometry.thickness=md.geometry.surface-md.geometry.bed
    md.geometry.base=md.geometry.surface-md.geometry.thickness

    md=setmask(md, '','')
    mask_pos=np.where(md.geometry.surface-md.geometry.bed>params['min_thickness_mask'])
    md.mask.ice_levelset=np.ones(md.mesh.numberofvertices)
    md.mask.ice_levelset[mask_pos]=-1

    #flat_bed=md.geometry.bed-(null_level+slope*md.mesh.x)
    #y_dim_friction=scale(-md.mesh.x)*5
    #md.friction.coefficient=flat_bed*0.03+params['friction']+y_dim_friction
    md.friction.coefficient=params['friction']*np.ones(md.mesh.numberofvertices)

    md=parameterize(md, 'my_Par2.py')
    md=setflowequation(md, 'SSA', 'all')

    md=SetMarineIceSheetBC(md)

    md.levelset.spclevelset=np.nan*np.ones(md.mesh.numberofvertices)
    md.levelset.spclevelset[np.where(md.geometry.bed>0)]=-1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    
    md.stressbalance.spcvx=np.nan*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvx[np.where(md.mesh.x<5)]=params['spcvx']
    md.stressbalance.spcvy=np.nan*np.ones(md.mesh.numberofvertices) 
    md.stressbalance.spcvy[md.mesh.vertexonboundary]=0
    #md.stressbalance.spcvy[np.where(md.mesh.x<2)]=np.nan
    #md.stressbalance.spcvy[np.where(np.logical_and(md.mesh.x<=x_dim, md.mesh.x>=x_dim-100))]=np.nan
    md.initialization.vx=params['spcvx']*np.ones(md.mesh.numberofvertices)
    md.initialization.vy=np.zeros(md.mesh.numberofvertices)
    md.initialization.vel=params['spcvx']*np.ones(md.mesh.numberofvertices)
    md.masstransport.min_thickness=1

    md.frontalforcings.meltingrate=params['frontal_melt']*np.ones(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate=params['floating_melt']*np.ones(md.mesh.numberofvertices)

    #rheology_B=cuffey(md.initialization.temperature)*1.1
    #md.materials.rheology_B=rheology_B*((md.mesh.y/(np.mean(md.mesh.y)))**0.85)
    #upper_half=np.where(md.mesh.y-10000>0.5*y_dim)
    #md.materials.rheology_B[upper_half]=rheology_B[upper_half]*(((-md.mesh.y[upper_half]+10000-params['bay_height1']+max(md.mesh.y))/(np.mean(-md.mesh.y+10000-params['bay_height1']+max(md.mesh.y))))**0.85)

    md.materials.rheology_B=cuffey(md.initialization.temperature)
    
    md.materials.rheology_law='Cuffey'

    md.calving.stress_threshold_groundedice=params['max_stress']
    md.calving.stress_threshold_floatingice=params['max_stress_floating'] 

    thk_dif=(dc+params['influx_height']+params['null_level'])*np.ones(len(md.mesh.x[np.where(md.mesh.x==0)]))-md.geometry.base[np.where(md.mesh.x<5)]
    thk_dif[np.where(thk_dif<0)]=1
    md.masstransport.spcthickness[np.where(md.mesh.x<5)]=thk_dif
    md.geometry.thickness[np.where(md.mesh.x<5)]=md.masstransport.spcthickness[np.where(md.mesh.x<5)]
    md.geometry.surface[np.where(md.mesh.x<5)]=md.geometry.base[np.where(md.mesh.x<5)]+md.geometry.thickness[np.where(md.mesh.x<5)]

    #md.transient.isgroundingline=0
    #md.transient.ismovingfront=0
    #md.calving.min_thickness=10
    return md



def SpinUp_load(params, run_name, load_name, fixfront):

    restart_time=-1
    y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()
    x_dim=params['x_dim']

    md=loadmodel(load_name) 
    md=transientrestart(md, md, restart_time)
    md.miscellaneous.name=run_name
    md.mask.ice_levelset[np.where(md.results.TransientSolution[restart_time].Thickness<=1)[0]]=1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    md.timestepping.final_time=params['final_time']
    md.timestepping.time_step=params['timestepping']
    md.settings.output_frequency=params['output_frequency']

    md.frontalforcings.meltingrate=params['frontal_melt']*np.ones(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate=params['floating_melt']*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvx[np.where(md.mesh.x<5)]=params['spcvx']
    
    md.calving=calvingvonmises()
    md.calving.stress_threshold_groundedice=params['max_stress']
    md.calving.stress_threshold_floatingice=params['max_stress_floating'] 
    md.friction.coefficient=params['friction']*np.ones(md.mesh.numberofvertices)

    if fixfront == True:
        md.transient.ismovingfront=0
    else:
        md.transient.ismovingfront=1


    md.smb.mass_balance=np.zeros(md.mesh.numberofvertices)
    md.smb.mass_balance[np.where(md.mesh.x<10000)]=params['smb']

    md.groundingline.melt_interpolation='FullMeltOnPartiallyFloating'
    #thk_dif=(dc+params['influx_height']+params['null_level'])*np.ones(len(md.mesh.x[np.where(md.mesh.x<5)]))-md.geometry.base[np.where(md.mesh.x<5)]
    #md.masstransport.spcthickness[np.where(md.mesh.x<5)]=thk_dif
    #md.geometry.thickness[np.where(md.mesh.x<5)]=md.masstransport.spcthickness[np.where(md.mesh.x<5)]
    #md.geometry.surface[np.where(md.mesh.x<5)]=md.geometry.base[np.where(md.mesh.x<5)]+md.geometry.thickness[np.where(md.mesh.x<5)]
    
    return md

def extenddomain(params, run_name, load_name, fixfront):
    restart_time=100
    y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()
    start_icefront=params['start_icefront']
    slab_thickness=params['slab_thickness']
    steepness=params['steepness']
    null_level=params['null_level']

    md2=loadmodel(load_name)

    x_dim=params['x_dim']
    generateEXP(params, x_dim, y_dim)
    md=bamg(model(), 'domain','exp_file2.exp', 'hmax',100, 'hmin', 100)

    md.geometry.bed=Geometry(md, y_dim+20000, x_dim, slope, params['bump_spread'], params['bump_height'], params['bump_pos'],params['bump_skew'], steepness, gap_halfwidth, dc, params['bay_spread1'], params['bay_height1'], params['bay_pos1'], params['bay_skew1'],params['bay_spread2'], params['bay_height2'], params['bay_pos2'], params['bay_skew2'],step, params['smb_pos'], params['funnel'])
    old_mesh_elements=md.mesh.elements
    old_mesh_x=md.mesh.x
    old_mesh_y=md.mesh.y
    old_mesh_geometry=md.geometry.bed
    h=np.nan*np.ones(md.mesh.numberofvertices)
    #h[np.where(np.logical_and(np.logical_and(md.mesh.y<17800, md.mesh.y>12200), np.logical_and(md.mesh.x<65000, md.mesh.x>20000)))]=100  #25000
    h[np.where(np.logical_and(md.geometry.bed<1200, np.logical_and(md.mesh.x<85000, md.mesh.x>30000)))]=100
    
    md=bamg(md, 'field', old_mesh_geometry, 'hmax', 1000, 'hmin', params['hmin'], 'gradation', 1, 'hVertices', h)
    md.miscellaneous.name=run_name

    md.geometry.bed=InterpFromMeshToMesh2d(old_mesh_elements, old_mesh_x, old_mesh_y, old_mesh_geometry, md.mesh.x, md.mesh.y)[0][:,0]+null_level
    
    md.geometry.thickness=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Thickness, md.mesh.x, md.mesh.y)[0][:,0]
    md.mask.groundedice_levelset=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].MaskGroundediceLevelset, md.mesh.x, md.mesh.y)[0][:,0]
    md.initialization.pressure=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Pressure, md.mesh.x, md.mesh.y)[0][:,0]
    md.geometry.base=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Base, md.mesh.x, md.mesh.y)[0][:,0]

    

    md.geometry.surface=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Surface, md.mesh.x, md.mesh.y)[0][:,0]
    md.geometry.surface[np.where(md.geometry.surface<md.geometry.bed)]=md.geometry.bed[np.where(md.geometry.surface<md.geometry.bed)]+1
    md.geometry.base[np.where(md.mask.groundedice_levelset>0)]=md.geometry.bed[np.where(md.mask.groundedice_levelset>0)]
    md.geometry.thickness[np.where(md.mask.groundedice_levelset>0)]=md.geometry.surface[np.where(md.mask.groundedice_levelset>0)]-md.geometry.base[np.where(md.mask.groundedice_levelset>0)]

    deep=np.where(md.geometry.base<md.geometry.bed)
    md.geometry.base[deep]=md.geometry.bed[deep]

    
    md.initialization.vx=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Vx, md.mesh.x, md.mesh.y)[0][:,0]
    md.initialization.vy=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Vy, md.mesh.x, md.mesh.y)[0][:,0]


    ## Parameterization
    md.smb.mass_balance=np.zeros(md.mesh.numberofvertices) 
    md.smb.mass_balance[np.where(md.mesh.x<10000)]=params['smb']
    md.calving=calvingvonmises()

    md.transient.isgroundingline=1
    md.transient.isthermal=1

    if fixfront == True:
        md.transient.ismovingfront=0
    else:
        md.transient.ismovingfront=1

    md.timestepping.start_time=0
    md.initialization.temperature=(273.15-5.)*np.ones((md.mesh.numberofvertices))
    md.materials.rheology_n=3*np.ones(md.mesh.numberofelements)
    md.friction.p=np.ones(md.mesh.numberofelements) 
    md.friction.q=np.ones(md.mesh.numberofelements) 
    md.basalforcings.groundedice_melting_rate=np.zeros(md.mesh.numberofvertices)

    grounded_mask=np.where(md.mask.groundedice_levelset>0)
    md.geometry.base[grounded_mask]=md.geometry.bed[grounded_mask]
    md.geometry.surface=md.geometry.base+md.geometry.thickness

    
    mask_pos=np.where(md.geometry.surface>1)
    md.mask.ice_levelset=np.ones(md.mesh.numberofvertices)
    md.mask.ice_levelset[mask_pos]=-1
    md.mask.ice_levelset[np.where(md.geometry.thickness>1)]=-1

    ##cutoff
    #md.mask.ice_levelset[np.where(md.mesh.x>90000)]=1
    #md.mask.ice_levelset[np.where(md.geometry.bed>1)]=-1
    #md.mask.ice_levelset[np.where(md.mask.groundedice_levelset<0)]=1
    ##
    
    md.friction.coefficient=params['friction']*np.ones(md.mesh.numberofvertices)
    
    md.timestepping.final_time=params['final_time']
    md.timestepping.time_step=params['timestepping']
    md.settings.output_frequency=params['output_frequency']

    md=setflowequation(md, 'SSA', 'all')

    md=SetMarineIceSheetBC(md)
    
    md.materials.rheology_B=cuffey(md.initialization.temperature)
    md.materials.rheology_law='Cuffey'

    md.levelset.spclevelset=np.nan*np.ones(md.mesh.numberofvertices)
    md.levelset.spclevelset[np.where(md.geometry.bed>0)]=-1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    
    md.stressbalance.spcvx=np.nan*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvx[np.where(md.mesh.x<5)]=params['spcvx']
    md.stressbalance.spcvy=np.nan*np.ones(md.mesh.numberofvertices) 
    md.stressbalance.spcvy[md.mesh.vertexonboundary]=0
    #md.stressbalance.spcvy[np.where(md.mesh.x==0)]=np.nan
   # md.stressbalance.spcvy[np.where(np.logical_and(md.mesh.x<=x_dim, md.mesh.x>=x_dim-100))]=np.nan

    md.masstransport.min_thickness=1

    md.frontalforcings.meltingrate=params['frontal_melt']*np.ones(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate=params['floating_melt']*np.ones(md.mesh.numberofvertices)
    md.cluster=md2.cluster
    
    md.calving.stress_threshold_groundedice=params['max_stress']
    md.calving.stress_threshold_floatingice=params['max_stress_floating'] 
    md.mask.ice_levelset[np.where(md.geometry.thickness<=10)]=1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    md.initialization.vx[np.where(md.geometry.thickness<=10)]=0
    md.initialization.vy[np.where(md.geometry.thickness<=10)]=0
    
    #thk_dif=(dc+params['influx_height']+params['null_level'])*np.ones(len(md.mesh.x[np.where(md.mesh.x<5)]))-md.geometry.base[np.where(md.mesh.x<5)]
    #thk_dif[np.where(thk_dif<0)]=1
    #md.masstransport.spcthickness[np.where(md.mesh.x<5)]=thk_dif
    #md.geometry.thickness[np.where(md.mesh.x<5)]=md.masstransport.spcthickness[np.where(md.mesh.x<5)]
    #md.geometry.surface[np.where(md.mesh.x<5)]=md.geometry.base[np.where(md.mesh.x<5)]+md.geometry.thickness[np.where(md.mesh.x<5)]
    md.groundingline.melt_interpolation='FullMeltOnPartiallyFloating'
    md=adddis(md, params)
    
    return md

def insertatinflux(params, run_name, load_name, fixfront):
    restart_time=50
    inpos=20000
    y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()
    start_icefront=params['start_icefront']
    slab_thickness=params['slab_thickness']
    steepness=params['steepness']
    null_level=params['null_level']

    md2=loadmodel(load_name)

    x_dim=params['x_dim']
    generateEXP(params, x_dim, y_dim)
    md=bamg(model(), 'domain','exp_file2.exp', 'hmax',100, 'hmin', 100)

    md.geometry.bed=Geometry(md, y_dim+20000, x_dim, slope, params['bump_spread'], params['bump_height'], params['bump_pos'],params['bump_skew'], steepness, gap_halfwidth, dc, params['bay_spread1'], params['bay_height1'], params['bay_pos1'], params['bay_skew1'],params['bay_spread2'], params['bay_height2'], params['bay_pos2'], params['bay_skew2'],step, params['smb_pos'], params['funnel'])
    old_mesh_elements=md.mesh.elements
    old_mesh_x=md.mesh.x
    old_mesh_y=md.mesh.y
    old_mesh_geometry=md.geometry.bed
    h=np.nan*np.ones(md.mesh.numberofvertices)
    #h[np.where(np.logical_and(np.logical_and(md.mesh.y<17800, md.mesh.y>12200), np.logical_and(md.mesh.x<65000, md.mesh.x>20000)))]=100  #25000
    h[np.where(np.logical_and(md.geometry.bed<1200, np.logical_and(md.mesh.x<90000, md.mesh.x>30000)))]=100

    md=bamg(md, 'field', old_mesh_geometry, 'hmax', 1000, 'hmin', params['hmin'], 'gradation', 1, 'hVertices', h)
    md.miscellaneous.name=run_name
    
    insert_over=np.where(md.mesh.x>inpos)
    insert_under=np.where(md.mesh.x<inpos)

    md.geometry.bed=InterpFromMeshToMesh2d(old_mesh_elements, old_mesh_x, old_mesh_y, old_mesh_geometry, md.mesh.x, md.mesh.y)[0][:,0]+null_level

    md.geometry.thickness=2000*np.ones(md.mesh.numberofvertices)
    md.geometry.thickness[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Thickness, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]
    #md.geometry.thickness[insert_under]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Thickness, md.mesh.x[insert_under], md.mesh.y[insert_under])[0][:,0]
    #md.geometry.thickness[insert_under]=np.mean(md2.results.TransientSolution[restart_time].Thickness[np.where(np.logical_and(md2.mesh.x<1000, np.logical_and(md2.mesh.y<17000, md2.mesh.y>13000)))])

    md.mask.groundedice_levelset=np.ones(md.mesh.numberofvertices)
    md.mask.groundedice_levelset[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].MaskGroundediceLevelset, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]

    md.initialization.pressure=np.ones(md.mesh.numberofvertices)
    md.initialization.pressure[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Pressure, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]
    md.initialization.pressure[insert_under]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Pressure, md.mesh.x[insert_under], md.mesh.y[insert_under])[0][:,0]

#    md.geometry.base=md.geometry.bed
    md.geometry.base=np.ones(md.mesh.numberofvertices)
    md.geometry.base[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Base, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]
    md.geometry.base[insert_under]=md.geometry.bed[insert_under]

    md.geometry.surface=2000*np.ones(md.mesh.numberofvertices)
    md.geometry.surface[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Surface, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]
    #md.geometry.surface[insert_under]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Surface, md.mesh.x[insert_under], md.mesh.y[insert_under])[0][:,0]
    md.geometry.surface[insert_under]=np.mean(md2.results.TransientSolution[restart_time].Surface[np.where(np.logical_and(md2.mesh.x<1000, np.logical_and(md2.mesh.y<17000, md2.mesh.y>13000)))])

    md.geometry.surface[np.where(md.geometry.surface<md.geometry.bed)]=md.geometry.bed[np.where(md.geometry.surface<md.geometry.bed)]+1
    md.geometry.base[np.where(md.mask.groundedice_levelset>0)]=md.geometry.bed[np.where(md.mask.groundedice_levelset>0)]
    md.geometry.thickness[np.where(md.mask.groundedice_levelset>0)]=md.geometry.surface[np.where(md.mask.groundedice_levelset>0)]-md.geometry.base[np.where(md.mask.groundedice_levelset>0)]

    deep=np.where(md.geometry.base<md.geometry.bed)
    md.geometry.base[deep]=md.geometry.bed[deep]


    md.initialization.vx=params['spcvx']*np.ones(md.mesh.numberofvertices)
    md.initialization.vx[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Vx, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]
    md.initialization.vy=np.zeros(md.mesh.numberofvertices)
    md.initialization.vy[insert_over]=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x+inpos, md2.mesh.y, md2.results.TransientSolution[restart_time].Vy, md.mesh.x[insert_over], md.mesh.y[insert_over])[0][:,0]

    md.smb.mass_balance=np.zeros(md.mesh.numberofvertices)
    md.smb.mass_balance[np.where(md.mesh.x<10000)]=params['smb']
    md.calving=calvingvonmises()

    md.transient.isgroundingline=1
    md.transient.isthermal=1
    if fixfront == True:
        md.transient.ismovingfront=0
    else:
        md.transient.ismovingfront=1

    md.timestepping.start_time=0
    md.initialization.temperature=(273.15-5.)*np.ones((md.mesh.numberofvertices))
    md.materials.rheology_n=3*np.ones(md.mesh.numberofelements)
    md.friction.p=np.ones(md.mesh.numberofelements)
    md.friction.q=np.ones(md.mesh.numberofelements)
    md.basalforcings.groundedice_melting_rate=np.zeros(md.mesh.numberofvertices)

    grounded_mask=np.where(md.mask.groundedice_levelset>0)
    md.geometry.base[grounded_mask]=md.geometry.bed[grounded_mask]
    md.geometry.surface=md.geometry.base+md.geometry.thickness


    mask_pos=np.where(md.geometry.surface>1)
    md.mask.ice_levelset=np.ones(md.mesh.numberofvertices)
    md.mask.ice_levelset[mask_pos]=-1
    md.mask.ice_levelset[np.where(md.geometry.thickness>1)]=-1

    ##cutoff
    #md.mask.ice_levelset[np.where(md.mesh.x>90000)]=1
    #md.mask.ice_levelset[np.where(md.geometry.bed>1)]=-1
    #md.mask.ice_levelset[np.where(md.mask.groundedice_levelset<0)]=1
    ##

    md.friction.coefficient=params['friction']*np.ones(md.mesh.numberofvertices)

    md.timestepping.final_time=params['final_time']
    md.timestepping.time_step=params['timestepping']
    md.settings.output_frequency=params['output_frequency']

    md=setflowequation(md, 'SSA', 'all')

    md=SetMarineIceSheetBC(md)

    md.materials.rheology_B=cuffey(md.initialization.temperature)
    md.materials.rheology_law='Cuffey'

    md.levelset.spclevelset=np.nan*np.ones(md.mesh.numberofvertices)
    md.levelset.spclevelset[np.where(md.geometry.bed>0)]=-1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1

    md.stressbalance.spcvx=np.nan*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvx[np.where(md.mesh.x<5)]=params['spcvx']
    md.stressbalance.spcvy=np.nan*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvy[md.mesh.vertexonboundary]=0
    #md.stressbalance.spcvy[np.where(md.mesh.x==0)]=np.nan
   # md.stressbalance.spcvy[np.where(np.logical_and(md.mesh.x<=x_dim, md.mesh.x>=x_dim-100))]=np.nan

    md.masstransport.min_thickness=1

    md.frontalforcings.meltingrate=params['frontal_melt']*np.ones(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate=params['floating_melt']*np.ones(md.mesh.numberofvertices)
    md.cluster=md2.cluster

    md.calving.stress_threshold_groundedice=params['max_stress']
    md.calving.stress_threshold_floatingice=params['max_stress_floating']
    md.mask.ice_levelset[np.where(md.geometry.thickness<=10)]=1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    md.initialization.vx[np.where(md.geometry.thickness<=10)]=0
    md.initialization.vy[np.where(md.geometry.thickness<=10)]=0

    #thk_dif=(dc+params['influx_height']+params['null_level'])*np.ones(len(md.mesh.x[np.where(md.mesh.x<5)]))-md.geometry.base[np.where(md.mesh.x<5)]
    #thk_dif[np.where(thk_dif<0)]=1
    #md.masstransport.spcthickness[np.where(md.mesh.x<5)]=thk_dif
    #md.geometry.thickness[np.where(md.mesh.x<5)]=md.masstransport.spcthickness[np.where(md.mesh.x<5)]
    #md.geometry.surface[np.where(md.mesh.x<5)]=md.geometry.base[np.where(md.mesh.x<5)]+md.geometry.thickness[np.where(md.mesh.x<5)]

    md=adddis(md, params)

    return md







































def remesh(params, run_name, load_name):
    restart_time=75
    y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()
    start_icefront=params['start_icefront']
    slab_thickness=params['slab_thickness']
    steepness=params['steepness']
    null_level=params['null_level']

    md2=loadmodel(load_name)

    x_dim=200000
    generateEXP(params, x_dim, y_dim)
    md=bamg(model(), 'domain','exp_file2.exp', 'hmax',100, 'hmin', 100)

    md.geometry.bed=Geometry(md, y_dim+20000, x_dim, slope, params['bump_spread'], params['bump_height'], params['bump_pos'], steepness, gap_halfwidth, dc, params['bay_spread1'], params['bay_height1'], params['bay_pos1'], params['bay_spread2'], params['bay_height2'], params['bay_pos2'], step)
    old_mesh_elements=md.mesh.elements
    old_mesh_x=md.mesh.x
    old_mesh_y=md.mesh.y
    old_mesh_geometry=md.geometry.bed
    
    md=bamg(md, 'field', md2.results.TransientSolution[restart_time].Vel, 'hmax', 1000, 'hmin', params['hmin'], 'gradation', 1.7)
    md.miscellaneous.name=run_name

    md.geometry.bed=InterpFromMeshToMesh2d(old_mesh_elements, old_mesh_x, old_mesh_y, old_mesh_geometry, md.mesh.x, md.mesh.y)[0][:,0]+null_level
    
    md.geometry.thickness=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Thickness, md.mesh.x, md.mesh.y)[0][:,0]
    #md.mask.ice_levelset=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[120].MaskIceLevelset, md.mesh.x, md.mesh.y)[0][:,0]
    md.mask.groundedice_levelset=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].MaskGroundediceLevelset, md.mesh.x, md.mesh.y)[0][:,0]
    md.initialization.pressure=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Pressure, md.mesh.x, md.mesh.y)[0][:,0]
    md.geometry.base=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Base, md.mesh.x, md.mesh.y)[0][:,0]

    deep=np.where(md.geometry.base<md.geometry.bed)
    md.geometry.base[deep]=md.geometry.bed[deep]
    
    
    md.initialization.vx=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Vx, md.mesh.x, md.mesh.y)[0][:,0]
    md.initialization.vy=InterpFromMeshToMesh2d(md2.mesh.elements, md2.mesh.x, md2.mesh.y, md2.results.TransientSolution[restart_time].Vy, md.mesh.x, md.mesh.y)[0][:,0]


    ## Parameterization
    #pressure_undefined=np.nonzero([not isinstance(x, float) for x in md.initialization.pressure]) 
    #md.initialization.pressure[pressure_undefined]=md.materials.rho_ice*md.constants.g*md.geometry.thickness
    md.smb.mass_balance=np.zeros(md.mesh.numberofvertices)
    md.smb.mass_balance=params['smb']*np.ones(md.mesh.numberofvertices)
    md.calving=calvingvonmises()

    md.transient.isgroundingline=1
    md.transient.isthermal=1
    md.transient.ismovingfront=1

    #Vel_undefined=np.nonzero([not isinstance(x, float) for x in md.initialization.vx])
    #md.initialization.vx[Vel_undefined]=0
    #md.initialization.vy[Vel_undefined]=0
    md.timestepping.start_time=0
    md.initialization.temperature=(273.15-5.)*np.ones((md.mesh.numberofvertices))
    md.materials.rheology_n=3*np.ones(md.mesh.numberofelements)
    md.friction.p=np.ones(md.mesh.numberofelements) 
    md.friction.q=np.ones(md.mesh.numberofelements) 
    md.basalforcings.groundedice_melting_rate=np.zeros(md.mesh.numberofvertices)

    #groundedice_undefined=np.nonzero([not isinstance(x, float) for x in md.mask.groundedice_levelset]) 
    #md.mask.groundedice_levelset[groundedice_undefined]=1

    grounded_mask=np.where(md.mask.groundedice_levelset>0)
    md.geometry.base[grounded_mask]=md.geometry.bed[grounded_mask]
    md.geometry.surface=md.geometry.base+md.geometry.thickness

    
    mask_pos=np.where(md.geometry.surface>1)
    md.mask.ice_levelset=np.ones(md.mesh.numberofvertices)
    md.mask.ice_levelset[mask_pos]=-1
    md.mask.ice_levelset[np.where(md.geometry.thickness>1)]=-1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1

    #flat_bed=md.geometry.bed-(null_level+slope*md.mesh.x)
    #y_dim_friction=scale(-md.mesh.x)*5
    #md.friction.coefficient=flat_bed*0.03+params['friction']+y_dim_friction
    md.friction.coefficient=params['friction']*np.ones(md.mesh.numberofvertices)

    md.timestepping.final_time=params['final_time']
    md.timestepping.time_step=params['timestepping']
    md.settings.output_frequency=params['output_frequency']

    md=setflowequation(md, 'SSA', 'all')

    md=SetMarineIceSheetBC(md)
    
    md.materials.rheology_B=cuffey(md.initialization.temperature)
    #md.materials.rheology_B=rheology_B*((md.mesh.y/(np.mean(md.mesh.y)))**0.85)
    #upper_half=np.where(md.mesh.y-10000>0.5*y_dim)
    #md.materials.rheology_B[upper_half]=rheology_B[upper_half]*(((-md.mesh.y[upper_half]+10000-params['bay_height1']+max(md.mesh.y))/(np.mean(-md.mesh.y+10000-params['bay_height1']+max(md.mesh.y))))**0.85)


    md.levelset.spclevelset=np.nan*np.ones(md.mesh.numberofvertices)
    md.levelset.spclevelset[np.where(md.geometry.bed>0)]=-1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1
    
    md.stressbalance.spcvx=np.nan*np.ones(md.mesh.numberofvertices)
    md.stressbalance.spcvx[np.where(md.mesh.x==0)]=params['spcvx']
    md.stressbalance.spcvy=np.nan*np.ones(md.mesh.numberofvertices) 
    md.stressbalance.spcvy[md.mesh.vertexonboundary]=0
    md.stressbalance.spcvy[np.where(md.mesh.x==0)]=np.nan
    md.stressbalance.spcvy[np.where(np.logical_and(md.mesh.x<=x_dim, md.mesh.x>=x_dim-100))]=np.nan
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1 

    md.masstransport.min_thickness=1

    md.frontalforcings.meltingrate=params['frontal_melt']*np.ones(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate=params['floating_melt']*np.ones(md.mesh.numberofvertices)
    md.cluster=md2.cluster
    md.mask.ice_levelset[np.where(md.results.TransientSolution[-1].Thickness<=50)[0]]=1
    md.mask.ice_levelset[np.where(md.levelset.spclevelset==-1)]=-1

    return md
