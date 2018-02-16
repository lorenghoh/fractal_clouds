from collections import OrderedDict
import json, pprint


def main():
    json_dict = OrderedDict()

    #---- BOMEX 
    json_dict['BOMEX'] = OrderedDict()
    json_dict['BOMEX']['config'] = OrderedDict()

    json_dict['BOMEX']['location'] = '/newtera/loh/data/BOMEX'

    stat_file = 'BOMEX_256x256x128_25m_25m_1s_stat.nc'

    json_dict['BOMEX']['condensed'] = '%s/condensed_entrain' % json_dict['BOMEX']['location']
    json_dict['BOMEX']['core'] = '%s/core_entrain' % json_dict['BOMEX']['location']
    json_dict['BOMEX']['stat_file'] = '%s/%s' % (json_dict['BOMEX']['location'], stat_file)
    json_dict['BOMEX']['tracking'] = '%s/tracking' % json_dict['BOMEX']['location']
    json_dict['BOMEX']['variables'] = '%s/variables' % json_dict['BOMEX']['location']

    # Model parameters
    json_dict['BOMEX']['config']['nx'] = 256
    json_dict['BOMEX']['config']['ny'] = 256
    json_dict['BOMEX']['config']['nz'] = 128
    json_dict['BOMEX']['config']['nt'] = 180

    json_dict['BOMEX']['config']['dx'] = 25
    json_dict['BOMEX']['config']['dt'] = 60

    json_dict['BOMEX']['config']['ug'] = -8.
    json_dict['BOMEX']['config']['vg'] = 0.


    #---- CGILS_300K
    json_dict['CGILS_300K'] = OrderedDict()
    json_dict['CGILS_300K']['config'] = OrderedDict()

    json_dict['CGILS_300K']['location'] = '/newtera/loh/data/CGILS_300K'

    stat_file = 'ENT_CGILS_S6_IDEAL_3D_SST_300K_384x384x194_25m_1s_stat.nc'
    
    json_dict['CGILS_300K']['condensed'] = '%s/condensed_entrain' % json_dict['CGILS_300K']['location']
    json_dict['CGILS_300K']['core'] = '%s/core_entrain' % json_dict['CGILS_300K']['location']
    json_dict['CGILS_300K']['stat_file'] = '%s/%s' % (json_dict['CGILS_300K']['location'], stat_file)
    json_dict['CGILS_300K']['tracking'] = '%s/tracking' % json_dict['CGILS_300K']['location']
    json_dict['CGILS_300K']['variables'] = '%s/variables' % json_dict['CGILS_300K']['location']

    # Model parameters
    json_dict['CGILS_300K']['config']['nt'] = 360
    json_dict['CGILS_300K']['config']['nx'] = 384
    json_dict['CGILS_300K']['config']['ny'] = 384
    json_dict['CGILS_300K']['config']['nz'] = 194

    json_dict['CGILS_300K']['config']['dx'] = 25
    json_dict['CGILS_300K']['config']['dt'] = 60

    json_dict['CGILS_300K']['config']['ug'] = 0.
    json_dict['CGILS_300K']['config']['vg'] = 0.


    #---- CGILS_301K
    json_dict['CGILS_301K'] = OrderedDict()
    json_dict['CGILS_301K']['config'] = OrderedDict()

    json_dict['CGILS_301K']['location'] = '/newtera/loh/data/CGILS_301K'

    stat_file = 'ENT_CGILS_S6_IDEAL_3D_SST_301K_384x384x194_25m_1s_stat.nc'
    
    json_dict['CGILS_301K']['condensed'] = '%s/condensed_entrain' % json_dict['CGILS_301K']['location']
    json_dict['CGILS_301K']['core'] = '%s/core_entrain' % json_dict['CGILS_301K']['location']
    json_dict['CGILS_301K']['stat_file'] = '%s/%s' % (json_dict['CGILS_301K']['location'], stat_file)
    json_dict['CGILS_301K']['tracking'] = '%s/tracking' % json_dict['CGILS_301K']['location']
    json_dict['CGILS_301K']['variables'] = '%s/variables' % json_dict['CGILS_301K']['location']

    # Model parameters
    json_dict['CGILS_301K']['config']['nt'] = 360
    json_dict['CGILS_301K']['config']['nx'] = 384
    json_dict['CGILS_301K']['config']['ny'] = 384
    json_dict['CGILS_301K']['config']['nz'] = 194

    json_dict['CGILS_301K']['config']['dx'] = 25
    json_dict['CGILS_301K']['config']['dt'] = 60

    json_dict['CGILS_301K']['config']['ug'] = 0.
    json_dict['CGILS_301K']['config']['vg'] = 0.


    #---- GCSSARM
    json_dict['GCSSARM'] = OrderedDict()
    json_dict['GCSSARM']['config'] = OrderedDict()

    json_dict['GCSSARM']['location'] = '/newtera/loh/data/GCSSARM'

    stat_file = 'GCSSARM_256x256x208_25m_25m_1s_stat.nc'

    json_dict['GCSSARM']['condensed'] = '%s/condensed_entrain' % json_dict['GCSSARM']['location']
    json_dict['GCSSARM']['core'] = '%s/core_entrain' % json_dict['GCSSARM']['location']
    json_dict['GCSSARM']['stat_file'] = '%s/%s' % (json_dict['GCSSARM']['location'], stat_file)
    json_dict['GCSSARM']['tracking'] = '%s/tracking' % json_dict['GCSSARM']['location']
    json_dict['GCSSARM']['variables'] = '%s/variables' % json_dict['GCSSARM']['location']

    # Model parameters
    json_dict['GCSSARM']['config']['nx'] = 256
    json_dict['GCSSARM']['config']['ny'] = 256
    json_dict['GCSSARM']['config']['nz'] = 128
    json_dict['GCSSARM']['config']['nt'] = 510

    json_dict['GCSSARM']['config']['dx'] = 25
    json_dict['GCSSARM']['config']['dt'] = 60

    json_dict['GCSSARM']['config']['ug'] = 10.
    json_dict['GCSSARM']['config']['vg'] = 0.


    #---- GATE_BDL
    json_dict['GATE_BDL'] = OrderedDict()
    json_dict['GATE_BDL']['config'] = OrderedDict()

    json_dict['GATE_BDL']['location'] = '/newtera/loh/data/GATE_BDL'

    stat_file = 'GATE_1920x1920x512_50m_1s_ent_stat.nc'

    json_dict['GATE_BDL']['condensed'] = '%s/condensed_entrain' % json_dict['GATE_BDL']['location']
    json_dict['GATE_BDL']['core'] = '%s/core_entrain' % json_dict['GATE_BDL']['location']
    json_dict['GATE_BDL']['stat_file'] = '%s/%s' % (json_dict['GATE_BDL']['location'], stat_file)
    json_dict['GATE_BDL']['tracking'] = '%s/tracking' % json_dict['GATE_BDL']['location']
    json_dict['GATE_BDL']['variables'] = '%s/variables' % json_dict['GATE_BDL']['location']

    # Model parameters
    json_dict['GATE_BDL']['config']['nx'] = 1728
    json_dict['GATE_BDL']['config']['ny'] = 1728
    json_dict['GATE_BDL']['config']['nz'] = 80
    json_dict['GATE_BDL']['config']['nt'] = 180

    json_dict['GATE_BDL']['config']['dx'] = 50
    json_dict['GATE_BDL']['config']['dt'] = 60

    json_dict['GATE_BDL']['config']['ug'] = -8.
    json_dict['GATE_BDL']['config']['vg'] = 0.


    #---- GATE
    json_dict['GATE'] = OrderedDict()
    json_dict['GATE']['config'] = OrderedDict()

    json_dict['GATE']['location'] = '/newtera/loh/data/GATE'

    stat_file = 'GATE_1920x1920x512_50m_1s_ent_stat.nc'

    json_dict['GATE']['condensed'] = '%s/condensed_entrain' % json_dict['GATE']['location']
    json_dict['GATE']['core'] = '%s/core_entrain' % json_dict['GATE']['location']
    json_dict['GATE']['stat_file'] = '%s/%s' % (json_dict['GATE']['location'], stat_file)
    json_dict['GATE']['tracking'] = '/tera/loh/cloudtracker/cloudtracker/hdf5'
    # json_dict['GATE']['tracking'] = '%s/tracking' % json_dict['GATE']['location']
    json_dict['GATE']['variables'] = '%s/variables' % json_dict['GATE']['location']

    # Model parameters
    json_dict['GATE']['config']['nx'] = 1728
    json_dict['GATE']['config']['ny'] = 1728
    json_dict['GATE']['config']['nz'] = 320
    json_dict['GATE']['config']['nt'] = 30

    json_dict['GATE']['config']['dx'] = 50
    json_dict['GATE']['config']['dt'] = 60

    json_dict['GATE']['config']['ug'] = -8.
    json_dict['GATE']['config']['vg'] = 0.


    with open('model_config.json','w') as f:
        json.dump(json_dict, f,indent=1)
        print('Wrote {} using util.write_json'.format('model_config.json'))
        pp = pprint.PrettyPrinter(indent=1)
        pp.pprint(json_dict)

if __name__ == '__main__':
    main()