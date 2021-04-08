from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_stage = namedtuple('Genotype_stage', 'cell cell_concat')
# Genotype_model = [Genotype_stage0, Genotype_stage1, Genotype_stage2, Genotype_stage3]

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# best_genotype
genotype_model = [
    Genotype_stage(cell=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_3x3', 2)], cell_concat=range(2, 6)),
    Genotype_stage(cell=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 4), ('avg_pool_3x3', 3)], cell_concat=range(2, 6)),
    Genotype_stage(cell=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], cell_concat=range(2, 6)),
    Genotype_stage(cell=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('skip_connect', 4), ('avg_pool_3x3', 1)], cell_concat=range(2, 6)),
    Genotype_stage(cell=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 4)], cell_concat=range(2, 6)),
    Genotype_stage(cell=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 4)], cell_concat=range(2, 6))
]