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


