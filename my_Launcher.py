from model import *
from solve import *
from os import path
from socket import gethostname
from export_netCDF import export_netCDF
from exportVTK import exportVTK
import my_Runner2
import math
from stallo import *
from generate_exp2 import *
from matplotlib.backends.backend_pdf import PdfPages
import glob

#Uncomment the platform to go on
#clustername = gethostname()
clustername = 'Stallo'


if clustername == 'Stallo':
    clusterident='Stallo'
    #cluster = stallo('numnodes', 1, 'cpuspernode', 16, 'time', 4 * 60, 'queue', 'devel')  # time in minutes
    cluster = stallo('numnodes', 1, 'cpuspernode', 16, 'mem', 1.9, 'time', 10 * 60)  #time in minutes
    #cluster = stallo('numnodes', 1, 'cpuspernode', 16, 'mem', 1.9, 'time', 10 * 60)  #time in minutes
else:
    #cluster = generic('name', clustername, 'np', 1, 'valgrind', '/usr/bin/valgrind', 'valgrindlib', '/usr/lib/valgrind/libmpiwrap-amd64-linux.so')
    cluster = generic('name', clustername, 'np', 10)
    clusterident='local'


prefix = 'GeomProj'

#run_type='SpinUp'   
#run_type='SpinUp_load' 
run_type='extenddomain' 
#run_type='insertatinflux'

plotting='off'

## Reference parameters
y_dim, x_dim, slope, dc, gap_halfwidth, step = standardvalues()

fixfront=False

bump_height=[0,0,0,0]
bump_spread=[0,0,0,0]
bump_pos=[0,0,0,0]
bump_skew=[0,0,0,0]

bay_height1=[900,-900,1800,-1800]
bay_spread1=[20000,20000,20000,20000]
bay_pos1=[55000,55000,55000,55000]
bay_skew1=[0,0,0,0]

bay_height2=[0,0,0,0]
bay_spread2=[0,0,0,0]
bay_pos2=[0,0,0,0]
bay_skew2=[0,0,0,0]

smb_pos=[30000]
funnel=[300]

smb=[55]
slab_thickness=[2000]
steepness=[1./300]
min_thickness_mask=[1]
spcvx=[50]
hmin=[350]
null_level=[-450]
frontal_melt=[200]
floating_melt=[30]
friction=[40]
start_icefront=[60000]
max_stress=[1000000]
max_stress_floating=[200000]
influx_height=[0]



final_time=[50]
timestepping=[0.01]
output_frequency=[100]

x_dim=[85000]
## define parameters lists

bump_height_list=[]
bump_spread_list=[]
bump_pos_list=[]
bump_skew_list=[]

bay_height1_list=[]
bay_spread1_list=[]
bay_pos1_list=[]
bay_skew1_list=[]

bay_height2_list=[]
bay_spread2_list=[]
bay_pos2_list=[]
bay_skew2_list=[]

smb_pos_list=[]
funnel_list=[]

smb_list=[]
slab_thickness_list=[]
steepness_list=[]
min_thickness_mask_list=[]
spcvx_list=[]
hmin_list=[]
null_level_list=[]
frontal_melt_list=[]
floating_melt_list=[]
friction_list=[]
start_icefront_list=[]

final_time_list=[]
timestepping_list=[]
output_frequency_list=[]
max_stress_list=[]
max_stress_floating_list=[]
x_dim_list=[]
influx_height_list=[]

for i, bump_height_val in enumerate(bump_height):
    #for q, bay_height1_val in enumerate(bay_height1):
    for slab_thickness_val in slab_thickness:
        for steepness_val in steepness:
            for min_thickness_mask_val in min_thickness_mask:
                for spcvx_val in spcvx:
                    for hmin_val in hmin:
                        for final_time_val in final_time:
                            for timestepping_val in timestepping:
                                for output_frequency_val in output_frequency:
                                    for null_level_val in null_level:
                                        for r,frontal_melt_val in enumerate(frontal_melt):
                                            for friction_val in friction:
                                                for start_icefront_val in start_icefront:
                                                    for max_stress_val in max_stress:
                                                        for max_stress_floating_val in max_stress_floating:
                                                            for x_dim_val in x_dim:
                                                                for influx_height_val in influx_height:
                                                                    for smb_pos_val in smb_pos:
                                                                        for funnel_val in funnel:
                                                                            for smb_val in smb:
                                                                                bump_height_list+=[bump_height_val]
                                                                                bump_spread_list+=[bump_spread[i]]
                                                                                bump_pos_list+=[bump_pos[i]]
                                                                                bump_skew_list+=[bump_skew[i]]
                                                                                bay_height1_list+=[bay_height1[i]]
                                                                                bay_spread1_list+=[bay_spread1[i]]
                                                                                bay_pos1_list+=[bay_pos1[i]]
                                                                                bay_skew1_list+=[bay_skew1[i]]
                                                                                bay_height2_list+=[bay_height2[i]]
                                                                                bay_spread2_list+=[bay_spread2[i]]
                                                                                bay_pos2_list+=[bay_pos2[i]]
                                                                                bay_skew2_list+=[bay_skew2[i]]
                                                                                slab_thickness_list+=[slab_thickness_val]
                                                                                steepness_list+=[steepness_val]
                                                                                min_thickness_mask_list+=[min_thickness_mask_val]
                                                                                spcvx_list+=[spcvx_val]
                                                                                hmin_list+=[hmin_val]
                                                                                final_time_list+=[final_time_val]
                                                                                timestepping_list+=[timestepping_val]
                                                                                output_frequency_list+=[output_frequency_val]
                                                                                null_level_list+=[null_level_val]
                                                                                frontal_melt_list+=[frontal_melt_val]
                                                                                floating_melt_list+=[floating_melt[r]]
                                                                                friction_list+=[friction_val]
                                                                                start_icefront_list+=[start_icefront_val]
                                                                                max_stress_list+=[max_stress_val]
                                                                                max_stress_floating_list+=[max_stress_floating_val]
                                                                                x_dim_list+=[x_dim_val]
                                                                                influx_height_list+=[influx_height_val]
                                                                                smb_pos_list+=[smb_pos_val]
                                                                                funnel_list+=[funnel_val]
                                                                                smb_list+=[smb_val]

run_number=len(bump_height_list)
 
launch_or_get = input('What do you want to do, launch (L), or retrieve (r): ')
#launch_or_get='r'

#[run_name, executable, restart_name, sollution] the run_name is also the executable
which_run={'SpinUp':['SpinUp', 'SpinUp', 'dummy', 'Transient'],
           'SetUp':['SetUp', 'SetUp', 'dummy', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_SG_spcvx2000_NL-500_FrM0_FlM0_FC60_FT200_TS0.05_OF20_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_SG_spcvx2000_NL-500_FrM0_FlM0_FC35_FT50_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_cleanStart_newBedFr.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_SG_spcvx2000_NL-500_FrM0_FlM0_FC32_FT50_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_cleanStart_newBedFr.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_load_SG_spcvx2100_NL-500_FrM0_FlM0_FC32_FT20_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_newBedFr.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_SG_spcvx500_NL-500_FrM0_FlM0_FC95_FT50_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_cleanStart_newBedFr.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM0_FlM0_FC95_FT30_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_newBedFr_MS1000000.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_SG_spcvx300_NL-500_FrM50_FlM10_FC50_FT20_TS0.0125_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_3DFr_MS1000000_newMassSpc_MSF300000_xdim200000.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_load_SG_spcvx300_NL-500_FrM200_FlM30_FC50_FT20_TS0.025_OF20_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_newBedFr_MS1000000_MassSpc_MSF300000.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_load_SG_spcvx300_NL-500_FrM200_FlM30_FC50_FT400_TS0.025_OF80_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_3DBedFr_MS1000000_newMassSpc_MSF300000_accT1700_FrGr5.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_SG_spcvx300_NL-500_FrM200_FlM30_FC50_FT100_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_3DFr_MS1000000_newMassSpc_MSF300000_xdim200000_FrGr5_fineMesh.nc','Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load', 'GeomProj_SpinUp_SG_spcvx300_NL-500_FrM200_FlM30_FC50_FT50_TS0.025_OF20_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_3DFr_MS1000000_newMassSpc_MSF200000_xdim200000_FrGr5_fineMesh.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx300_NL-500_FrM400_FlMreal50_FC150_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_accT100.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_parStart40_accT100.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_SG_spcvx800_NL-500_FrM400_FlMreal50_FC150_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newestMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_parStart10.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF100000_xdim60000_fineMesh_noMinCalv_parStart10.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT100_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim60000_fineMesh_noMinCalv_accT150_parStart10.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx800_NL-500_FrM400_FlMreal50_FC40_FT30_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim60000_fineMesh_noMinCalv_accT180_parStart10.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx800_NL-500_FrM400_FlMreal50_FC40_FT20_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim60000_fineMesh_noMinCalv_accT200_parStart10.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC40_FT30_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim60000_fineMesh_noMinCalv_accT250_parStart10.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC40_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim60000_fineMesh_noMinCalv_accT300_parStart10.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC40_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim60000_fineMesh_noMinCalv_accT350_parStart10.nc', 'Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC38_FT10_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim60000_fineMesh_noMinCalv_accT360_parStart10.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC38_FT20_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim60000_fineMesh_noMinCalv_accT380_parStart10_inH500.nc','Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC36_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF500000_xdim70000_fineMesh_noMinCalv_accT450_parStart10_inH500.nc','Transient'],
            #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC36_FT30_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF500000_xdim70000_fineMesh_noMinCalv_accT480_parStart10_inH500.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT20_TS0.01_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim110000_fineMesh_noMinCalv_accT520_parStart10_inH500.nc','Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT30_TS0.01_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim110000_fineMesh_noMinCalv_accT550_parStart10_inH500.nc', 'Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT20_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF200000_xdim40000_fineMesh_noMinCalv_parStart10.nc', 'Transient'],###
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal50_FC50_FT150_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim105000_fineMesh_noMinCalv_accT350_parStart35_inH0.nc','Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM400_FlMreal50_FC50_FT100_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim105000_fineMesh_noMinCalv_accT450_parStart35_inH0.nc','Transient'],
          # 'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF250000_xdim105000_fineMesh_noMinCalv_accT750_parStart35_inH0.nc','Transient'],
          # 'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_insertatinflux_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT200_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT225_inH0_smb55_smbPos30000_funnel300.nc','Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_insertatinflux_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT250_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT225_inH0_smb53_smbPos30000_funnel300.nc','Transient'],
           #'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT200_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT575_inH0_smb55_smbPos30000_funnel300.nc','Transient'],
          # 'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT675_inH0_smb55_smbPos30000_funnel300.nc','Transient'],
           'SpinUp_load':['SpinUp_load', 'SpinUp_load','GeomProj_SpinUp_load_SG_spcvx50_NL-450_FrM800_FlMreal120_FC40_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT725_inH0_smb55_smbPos30000_funnel300_FullMelt.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_load_SG_spcvx2000_NL-500_FrM0_FlM0_FC35_FT200_TS0.025_OF40_hmin250_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_linRheol_newBedFr.nc', 'Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx300_NL-500_FrM200_FlM30_FC50_FT100_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_linRheol_3DFr_MS1000000_newMassSpc_MSF300000_xdim200000_FrGr5_fineMesh.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_SG_spcvx300_NL-500_FrM400_FlMreal50_FC50_FT20_TS0.025_OF20_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newMassSpc_MSF200000_xdim200000_fineMesh.nc', 'Transient'],
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_load_SG_spcvx300_NL-500_FrM400_FlMreal50_FC150_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_accT100.nc', 'Transient'],
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newerMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_parStart40.nc', 'Transient'],
            #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_load_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS1000000_newMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_parStart40_accT150.nc', 'Transient'],
            #'extenddomain':['extenddomain', 'extenddomain',  'GeomProj_SpinUp_SG_spcvx600_NL-500_FrM400_FlMreal50_FC30_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF150000_MS4000000_newestMassSpc_MSF100000_xdim160000_fineMesh_noMinCalv_parStart10.nc', 'Transient'],
           #'extenddomain':['extenddomain', 'extenddomain',  'GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC40_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim60000_fineMesh_noMinCalv_accT300_parStart10.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC38_FT20_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim60000_fineMesh_noMinCalv_accT400_parStart10_inH500.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx1000_NL-500_FrM400_FlMreal50_FC36_FT20_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF500000_xdim70000_fineMesh_noMinCalv_accT500_parStart10_inH500.nc','Transient']}
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT10_TS0.01_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF700000_xdim70000_fineMesh_noMinCalv_accT500_parStart10_inH500.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT20_TS0.0125_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF600000_xdim90000_fineMesh_noMinCalv_accT520_parStart10_inH500.nc','Transient'],
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM400_FlMreal50_FC36_FT30_TS0.0125_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF200000_xdim110000_fineMesh_noMinCalv_accT580_parStart10_inH500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx500_NL-500_FrM400_FlMreal50_FC50_FT100_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF400000_xdim70000_fineMesh_noMinCalv_parStart35.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim80000_fineMesh_noMinCalv_accT150_parStart35_inH0.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal50_FC50_FT50_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim90000_fineMesh_noMinCalv_accT167_parStart35_inH0.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal50_FC50_FT150_TS0.0125_OF80_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim80000_fineMesh_noMinCalv_accT350_parStart35_inH0.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_SpinUp_load_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF400000_xdim105000_fineMesh_noMinCalv_accT550_parStart35_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain', 'GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim115000_fineMesh_noMinCalv_accT650_parStart35_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain',  'GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF250000_xdim110000_fineMesh_noMinCalv_accT700_parStart35_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain',  'GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.025_OF40_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF250000_xdim105000_fineMesh_noMinCalv_accT750_parStart35_inH0.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT10_TS0.02_OF10_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF200000_xdim105000_fineMesh_noMinCalv_accT7800_parStart35_inH0.nc', 'Transient']
           # 'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx500_NL-500_FrM400_FlMreal20_FC50_FT10_TS0.01_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newestMassSpc_MSF200000_xdim100000_fineMesh_noMinCalv_parStart10.nc', 'Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx600_NL-500_FrM400_FlMreal20_FC45_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newestMassSpc_MSF300000_xdim95000_fineMesh75_noMinCalv_parStart10.nc', 'Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx300_NL-500_FrM400_FlMreal20_FC55_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newestMassSpc_MSF245000_xdim95000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx400_NL-500_FrM400_FlMreal20_FC55_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim105000_finestMesh_noMinCalv_accT100_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx400_NL-500_FrM400_FlMreal20_FC55_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim98000_finestMesh_noMinCalv_accT250_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim98000_finestMesh_noMinCalv_accT300_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim100000_finestMesh_noMinCalv_accT350_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF230000_xdim100000_finestMesh_noMinCalv_accT370_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim100000_finestMesh_noMinCalv_accT370_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF200000_xdim100000_finestMesh_noMinCalv_accT400_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF170000_xdim100000_finestMesh_noMinCalv_accT450_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx600_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF140000_xdim100000_finestMesh_noMinCalv_accT470_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM400_FlMreal20_FC55_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim100000_finestMesh_noMinCalv_accT500_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM400_FlMreal20_FC55_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT530_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM400_FlMreal20_FC50_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT550_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx800_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT550_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx1200_NL-500_FrM400_FlMreal20_FC55_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT580_parStart10_inH0.nc', 'Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx1200_NL-500_FrM400_FlMreal20_FC55_FT20_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT520_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx1200_NL-500_FrM400_FlMreal20_FC55_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF150000_xdim95000_finestMesh_noMinCalv_accT550_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx300_NL-500_FrM400_FlMreal20_FC120_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newestMassSpc_MSF150000_xdim95000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx400_NL-500_FrM400_FlMreal20_FC60_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newMassSpc_MSF245000_xdim98000_finestMesh_noMinCalv_accT300_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx400_NL-500_FrM400_FlMreal20_FC70_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF150000_xdim30000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx1000_NL-500_FrM400_FlMreal20_FC70_FT30_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF150000_xdim30000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx1500_NL-500_FrM400_FlMreal20_FC50_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF150000_xdim30000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx1500_NL-500_FrM400_FlMreal20_FC70_FT50_TS0.02_OF50_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF150000_xdim30000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT20_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF300000_xdim58000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT20_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim58000_finestMesh_noMinCalv_accT40_parStart10_inH0.nc','Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim62000_finestMesh_noMinCalv_accT140_parStart10_inH0.nc','Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT20_TS0.01_OF100_hmin350_BuH300_BuP40000_BuS20000_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim66000_finestMesh_noMinCalv_accT157_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT20_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH2500_ByP40000_ByS20000_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim66000_finestMesh_noMinCalv_accT157_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT30_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim66000_finestMesh_noMinCalv_accT157_parStart10_inH0.nc','Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx5000_NL-500_FrM400_FlMreal20_FC30_FT30_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH2500_ByP40000_ByS20000_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim66000_finestMesh_noMinCalv_accT187_parStart10_inH0.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx3500_NL-500_FrM1000_FlMreal60_FC35_FT25_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF300000_xdim50000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx3500_NL-500_FrM1000_FlMreal60_FC35_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim70000_finestMesh_noMinCalv_accT122_parStart10_inH500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx3500_NL-500_FrM1000_FlMreal60_FC35_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim78000_finestMesh_noMinCalv_accT139_parStart10_inH500.nc','Transient']
           # 'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx3500_NL-500_FrM1000_FlMreal60_FC35_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim70000_finestMesh_noMinCalv_accT190_parStart10_inH500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx3500_NL-500_FrM1000_FlMreal60_FC35_FT30_TS0.01_OF100_hmin350_BuH150_BuP40000_BuS40000_ByH1000_ByP40000_ByS31250_Stallo_IF20000_MS1000000_newMassSpc_MSF200000_xdim70000_finestMesh_noMinCalv_accT220_parStart10_inH500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx3500_NL-500_FrM1300_FlMreal90_FC35_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim70000_finestMesh_noMinCalv_accT290_parStart10_inH500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM1000_FlMreal60_FC35_FT30_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim70000_finestMesh_noMinCalv_accT220_parStart10_inH1500.nc','Transient'] #REF
          #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx400_NL-500_FrM1000_FlMreal60_FC35_FT80_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF300000_xdim70000_finestMesh_noMinCalv_accT300_parStart10_inH1500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx400_NL-500_FrM1000_FlMreal60_FC35_FT20_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim70000_finestMesh_noMinCalv_accT320_parStart10_inH1500.nc','Transient']
            #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx700_NL-500_FrM0_FlMreal0_FC35_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newestMassSpc_MSF100000_xdim50000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM0_FlMreal0_FC35_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim70000_finestMesh_noMinCalv_accT127_parStart10_inH1500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM0_FlMreal0_FC35_FT50_TS0.01_OF100_hmin350_BuH300_BuP30000_BuS40000_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim70000_finestMesh_noMinCalv_accT177_parStart10_inH1500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx700_NL-500_FrM0_FlMreal0_FC35_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF20000_MS1000000_newMassSpc_MSF100000_xdim70000_finestMesh_noMinCalv_accT177_parStart10_inH1500.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx300_NL-500_FrM200_FlMreal30_FC50_FT25_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF90000_MS1000000_newestMassSpc_MSF100000_xdim92000_fineMesh75_noMinCalv_parStart10.nc','Transient']
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_SG_spcvx800_NL-500_FrM200_FlMreal30_FC40_FT125_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_newestMassSpc_MSF100000_xdim62000_fineMesh75_noMinCalv_parStart2.nc','Transient']
          # 'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx800_NL-500_FrM200_FlMreal30_FC40_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_newMassSpc_MSF200000_xdim65000_finestMesh_noMinCalv_accT175_parStart10_inH0_fixedfrontandnocal.nc','Transient'],
          # 'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx50_NL-450_FrM200_FlMreal100_FC40_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT825_inH0_smb55_smbPos30000_funnel300_FullMelt.nc','Transient'],
           'extenddomain':['extenddomain', 'extenddomain','GeomProj_SpinUp_load_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT875_inH0_smb55_smbPos30000_funnel300_FullMelt.nc','Transient'],
          # 'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH-900_ByP55000_ByS20000_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT925_inH0_smb55_smbPos30000_funnel300_FullMelt_noCutoff.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH-675_ByP55000_ByS20000_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT925_inH0_smb55_smbPos30000_funnel300_FullMelt_noCutoff.nc','Transient'],
           #'extenddomain':['extenddomain', 'extenddomain','GeomProj_extenddomain_SG_spcvx50_NL-450_FrM200_FlMreal30_FC40_FT100_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH1350_ByP55000_ByS20000_Stallo_IF60000_MS1000000_MSF200000_xdim85000_accT925_inH0_smb55_smbPos30000_funnel300_FullMelt_noCutoff.nc','Transient'],
           'insertatinflux':['insertatinflux', 'insertatinflux', 'GeomProj_extenddomain_SG_spcvx800_NL-500_FrM200_FlMreal30_FC40_FT50_TS0.01_OF100_hmin350_BuH0_BuP0_BuS0_ByH0_ByP0_ByS0_Stallo_IF60000_MS1000000_newMassSpc_MSF200000_xdim65000_finestMesh_noMinCalv_accT175_parStart10_inH0_fixedfrontandnocal.nc','Transient']
           }


## Define parameter Dict
if launch_or_get == 'r' and plotting=='on':
    pp=PdfPages('runplots_Friction_Speed_Balance_newBedFr_thirties.pdf')
for run in range(0,run_number):
    params={'bump_height': bump_height_list[run],
            'bump_spread': bump_spread_list[run],
            'bump_pos': bump_pos_list[run],
            'bump_skew': bump_skew_list[run],
            'bay_height1': bay_height1_list[run],
            'bay_spread1': bay_spread1_list[run],
            'bay_pos1': bay_pos1_list[run],
            'bay_skew1':bay_skew1_list[run],
            'bay_height2': bay_height2_list[run],
            'bay_spread2': bay_spread2_list[run],
            'bay_pos2': bay_pos2_list[run],
            'bay_skew2': bay_skew2_list[run],
            'slab_thickness': slab_thickness_list[run],
            'steepness': steepness_list[run],
            'min_thickness_mask': min_thickness_mask_list[run],
            'spcvx': spcvx_list[run],
            'hmin': hmin_list[run],
            'final_time': final_time_list[run],
            'timestepping': timestepping_list[run],
            'output_frequency': output_frequency_list[run],
            'null_level': null_level_list[run],
            'frontal_melt': frontal_melt_list[run],
            'floating_melt': floating_melt_list[run],
            'friction': friction_list[run],
            'start_icefront':start_icefront_list[run],
            'max_stress':max_stress_list[run],
            'max_stress_floating':max_stress_floating_list[run],
            'x_dim':x_dim_list[run],
            'influx_height':influx_height_list[run],
            'smb_pos': smb_pos_list[run],
            'funnel': funnel_list[run],
            'smb': smb_list[run]}
    
    if run_type == 'SpinUp':
        load_name=which_run[run_type][2]+'.nc'
        #run_name=prefix+'_'+which_run[run_type][0]+str(slab_thickness_list[run])+'_'+str(final_time_list[run])+'a'+'_'+'steepness'+str(round(steepness_list[run],3)) +'_'+'min_thickness_mask'+str(min_thickness_mask_list[run])+'_'+'timestep'+str(timestepping_list[run])+'_'+'hmin'+str(hmin_list[run])+'_'+'gradation1.7'+'_'+'spcvx'+str(spcvx_list[run])+'_'+'hmax1000'+'_'+'ydim20000'+'gap_halfwidth2500'+'_testnewshape'
        if bay_pos1==bay_pos2:
            run_name=prefix+'_'+which_run[run_type][0]+'_SG'+'_spcvx'+str(spcvx_list[run])+'_NL'+str(params['null_level'])+'_FrM'+str(params['frontal_melt'])+'_FlMreal'+str(params['floating_melt'])+'_FC'+str(params['friction'])+'_FT'+str(params['final_time'])+'_TS'+str(params['timestepping'])+'_OF'+str(params['output_frequency'])+'_hmin'+str(params['hmin'])+'_BuH'+str(params['bump_height'])+'_BuP'+str(params['bump_pos'])+'_BuS'+str(params['bump_spread'])+'_ByH'+str(params['bay_height1'])+'_ByP'+str(params['bay_pos1'])+'_ByS'+str(params['bay_spread1'])+'_'+clusterident+'_IF'+str(params['start_icefront'])+'_MS'+str(params['max_stress'])+'_newestMassSpc'+'_MSF'+str(params['max_stress_floating'])+'_xdim'+str(params['x_dim'])+'_fineMesh75'+'_noMinCalv'+'_parStart2'
        else:
            run_name=prefix+'_'+which_run[run_type][0]+'_spcvx'+str(spcvx_list[run])+'_NL'+str(params['null_level'])+'_FrM'+str(params['frontal_melt'])+'_FlM'+str(params['floating_melt'])+'_FT'+str(params['final_time'])+'_TS'+str(params['timestepping'])+'_OF'+str(params['output_frequency'])+'_hmin'+str(params['hmin'])+'_BuH'+str(params['bump_height'])+'_BuP'+str(params['bump_pos'])+'_BuS'+str(params['bump_spread'])+'_ByH1'+str(params['bay_height1'])+'_ByP1'+str(params['bay_pos1'])+'_ByS1'+str(params['bay_spread1'])+'_ByH2'+str(params['bay_height2'])+'_ByP2'+str(params['bay_pos2'])+'_ByS2'+str(params['bay_pos2'])+'_'+clusterident+'_IF'+str(params['start_icefront'])+'_linRheol'+'spcvy0'
    elif run_type == 'SetUp':
        load_name=which_run[run_type][2]+'.nc'
        run_name=prefix+'_'+which_run[run_type][0]+str(slab_thickness_list[run])+'_'+str(final_time_list[run])+'a'+'_'+'steepness'+str(round(steepness_list[run],3)) +'_'+'min_thickness_mask'+str(min_thickness_mask_list[run])+'_'+'timestep'+str(timestepping_list[run])+'_'+'hmin'+str(hmin_list[run])+'_'+'gradation1.7'+'_'+'spcvx'+str(spcvx_list[run])+'_2ndtry'
    elif run_type=='SpinUp_load':
        load_name='../../share_setup/Models/'+which_run[run_type][2]
        run_name=prefix+'_'+which_run[run_type][0]+'_SG'+'_spcvx'+str(spcvx_list[run])+'_NL'+str(params['null_level'])+'_FrM'+str(params['frontal_melt'])+'_FlMreal'+str(params['floating_melt'])+'_FC'+str(params['friction'])+'_FT'+str(params['final_time'])+'_TS'+str(params['timestepping'])+'_OF'+str(params['output_frequency'])+'_hmin'+str(params['hmin'])+'_BuH'+str(params['bump_height'])+'_BuP'+str(params['bump_pos'])+'_BuS'+str(params['bump_spread'])+'_ByH'+str(params['bay_height1'])+'_ByP'+str(params['bay_pos1'])+'_ByS'+str(params['bay_spread1'])+'_'+clusterident+'_IF'+str(params['start_icefront'])+'_MS'+str(params['max_stress'])+'_MSF'+str(params['max_stress_floating'])+'_xdim'+str(params['x_dim'])+'_accT825'+'_inH'+str(params['influx_height'])+'_smb'+str(params['smb'])+'_smbPos'+str(params['smb_pos'])+'_funnel'+str(params['funnel'])+'_FullMelt'
    elif run_type == 'extenddomain':
        load_name='../../share_setup/Models/'+which_run[run_type][2]
        run_name=prefix+'_'+which_run[run_type][0]+'_SG'+'_spcvx'+str(spcvx_list[run])+'_NL'+str(params['null_level'])+'_FrM'+str(params['frontal_melt'])+'_FlMreal'+str(params['floating_melt'])+'_FC'+str(params['friction'])+'_FT'+str(params['final_time'])+'_TS'+str(params['timestepping'])+'_OF'+str(params['output_frequency'])+'_hmin'+str(params['hmin'])+'_BuH'+str(params['bump_height'])+'_BuP'+str(params['bump_pos'])+'_BuS'+str(params['bump_spread'])+'_ByH'+str(params['bay_height1'])+'_ByP'+str(params['bay_pos1'])+'_ByS'+str(params['bay_spread1'])+'_'+clusterident+'_IF'+str(params['start_icefront'])+'_MS'+str(params['max_stress'])+'_MSF'+str(params['max_stress_floating'])+'_xdim'+str(params['x_dim'])+'_accT925'+'_inH'+str(params['influx_height'])+'_smb'+str(params['smb'])+'_smbPos'+str(params['smb_pos'])+'_funnel'+str(params['funnel'])+'_FullMelt'+'_noCutoff'+'_asymmetric'
    elif run_type=='insertatinflux':
        load_name='../../share_setup/Models/'+which_run[run_type][2]
        run_name=prefix+'_'+which_run[run_type][0]+'_SG'+'_spcvx'+str(spcvx_list[run])+'_NL'+str(params['null_level'])+'_FrM'+str(params['frontal_melt'])+'_FlMreal'+str(params['floating_melt'])+'_FC'+str(params['friction'])+'_FT'+str(params['final_time'])+'_TS'+str(params['timestepping'])+'_OF'+str(params['output_frequency'])+'_hmin'+str(params['hmin'])+'_BuH'+str(params['bump_height'])+'_BuP'+str(params['bump_pos'])+'_BuS'+str(params['bump_spread'])+'_ByH'+str(params['bay_height1'])+'_ByP'+str(params['bay_pos1'])+'_ByS'+str(params['bay_spread1'])+'_'+clusterident+'_IF'+str(params['start_icefront'])+'_MS'+str(params['max_stress'])+'_MSF'+str(params['max_stress_floating'])+'_xdim'+str(params['x_dim'])+'_accT225'+'_inH'+str(params['influx_height'])+'_smb'+str(params['smb'])+'_smbPos'+str(params['smb_pos'])+'_funnel'+str(params['funnel'])

    else:
        print('run_type "{}" is not recognised possibilities are :{}'.format(run_type, which_run.keys()))
    
    # }}}
    # Check existence {{{
    #vtk_name = '/media/thomas/TOSHIBA EXT/Geometry_Project/Results/' + run_name
    model_name= '../../share_setup/Models/'+run_name + '.nc'
    if path.exists(model_name):
        print('File {} allready exists, skipping'.format(model_name))
        continue
    #if not path.exists(load_name):
    #    print('Load name {} does not exist, skipping'.format(load_name))
    #    continue
    if path.exists('./' + run_name + '.bin') and launch_or_get == 'L':
        print('{} allready runing, skipping'.format(run_name))
        continue
    #if not path.exists('./' + run_name + '.bin') and launch_or_get == 'r' and clustername != gethostname():
     #   print('{} has no local bin,  skipping'.format(run_name))
     #   continue





    if launch_or_get == 'L':
        print('Launching with file {}'.format(load_name))
        writelog(params, run_name, x_dim, y_dim, dc, gap_halfwidth, slope, steepness, step, clusterident, 'log_L.csv', 'L')
        md = getattr(my_Runner2, which_run[run_type][1])(params, run_name, load_name, fixfront)
        md.cluster = cluster
        if clustername != gethostname():
            md.cluster.interactive = 0
            md.settings.waitonlock = 0
        else:
            md.cluster.interactive = 1
            md.settings.waitonlock = math.inf
        if fixfront==True:
            md.transient.requested_outputs=['TotalSmb','SmbMassBalance','IceVolume','IceVolumeAboveFloatation',  'IceVolumeAboveFloatationScaled','GroundedAreaScaled',  'FloatingAreaScaled','IceMass','GroundedArea','FloatingArea','TotalFloatingBmb',   'BasalforcingsFloatingiceMeltingRate', 'IcefrontMassFluxLevelset','IcefrontMassFlux', 'GroundinglineMassFlux']
        else:
            md.transient.requested_outputs=['TotalSmb','SmbMassBalance','IceVolume','IceVolumeAboveFloatation',  'IceVolumeAboveFloatationScaled','GroundedAreaScaled',  'FloatingAreaScaled','IceMass','GroundedArea','FloatingArea','TotalFloatingBmb',   'BasalforcingsFloatingiceMeltingRate', 'IcefrontMassFluxLevelset','IcefrontMassFlux', 'GroundinglineMassFlux', 'TotalCalvingFluxLevelset', 'CalvingMeltingFluxLevelset']
        md = solve(md, which_run[run_type][3], 'runtimename', 0)
        
    # }}}
    # Retriever {{{
    elif launch_or_get == 'r':
        print('retreiving')
        md = getattr(my_Runner2, which_run[run_type][1])(params, run_name, load_name, fixfront)
        md.cluster = cluster
        md = loadresultsfromcluster(md, run_name)
        export_netCDF(md, model_name)
        #exportVTK(vtk_name, md, 'geometry', 'mesh','mask',  'stressbalance', 'materials', 'friction', 'calving')
        filename = md.miscellaneous.name
        if len(md.results.TransientSolution)==(md.timestepping.final_time*10/md.settings.output_frequency)+1:
            writelog(params, run_name, x_dim, y_dim, dc, gap_halfwidth, slope, steepness, step, clusterident, 'log_R.csv', 'S')
        else:
            writelog(params, run_name, x_dim, y_dim, dc, gap_halfwidth, slope, steepness, step, clusterident, 'log_R.csv', 'F')

        if plotting =='on':           
            r=np.linspace(0,len(md.results.TransientSolution),10)
            r = [ int(x) for x in r ]
            if max(r)<=10:
                   pp.savefig(plotmodel(md, 'data', md.results.TransientSolution[0].Vel, 'data', md.results.TransientSolution[-1].Vel, 'colorbar', 'off', 'title', md.miscellaneous.name+'_Vel'))
                   pp.savefig(plotmodel(md, 'data', md.results.TransientSolution[0].Thickness, 'data', md.results.TransientSolution[-1].Thickness, 'colorbar', 'off', 'title', md.miscellaneous.name+'_Thk'))
            else:
                pp.savefig(plotmodel(md, 'data', md.results.TransientSolution[r[0]].Vel,'data', md.results.TransientSolution[r[1]].Vel, 'data',md.results.TransientSolution[r[3]].Vel,'data',md.results.TransientSolution[r[4]].Vel, 'data',md.results.TransientSolution[r[5]].Vel, 'data',md.results.TransientSolution[r[6]].Vel,'data',md.results.TransientSolution[r[7]].Vel, 'data',md.results.TransientSolution[r[8]].Vel, 'data',md.results.TransientSolution[-1].Vel, 'colorbar', 'off', 'title#2', md.miscellaneous.name+'\n_Vel(%s%s%s%s%s%s%s)'%(r[1], r[3], r[4],r[5], r[6], r[7],r[8])))
                pp.savefig(plotmodel(md, 'data', md.results.TransientSolution[0].Thickness, 'data',md.results.TransientSolution[r[1]].Thickness, 'data',md.results.TransientSolution[r[3]].Thickness,'data',md.results.TransientSolution[r[4]].Thickness, 'data',md.results.TransientSolution[r[5]].Thickness, 'data',md.results.TransientSolution[r[6]].Thickness, 'data',md.results.TransientSolution[r[7]].Thickness, 'data',md.results.TransientSolution[r[8]].Thickness,'data',md.results.TransientSolution[-1].Thickness, 'colorbar', 'off', 'title#2', md.miscellaneous.name+'\n_Thk(%s%s%s%s%s%s%s)'%(r[1], r[3], r[4],r[5], r[6], r[7],r[8])))
        for extension in ['.bin', '.queue', '.toolkits']:
            try:
                os.remove(filename + extension)
            except OSError:
                print('WARNING,  no ' + extension + '  is present for run ' + filename)
    # }}}
    else:
        print('Bad Entry, Try again')

if launch_or_get=='r' and plotting =='on':
    pp.close()

 
 
