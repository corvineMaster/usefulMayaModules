"""
A collection of utility functions for generating most of Maya's math nodes, especially the ones created in 2024 and 2025 due to their standardized plug names.
"""
from typing import Union
import maya.cmds as cmds


########## Universal helper functions ##########

def _create_multi_input_math_node(node_type: str, inputs: list[Union[str, int, float]], targets: list[str]=None, matrix: bool=False):
    if not isinstance(inputs, tuple) and not isinstance(inputs, list):
        inputs = [inputs]
    if targets is None:
        targets = []

    node = cmds.createNode(node_type)
    for i, input in enumerate(inputs):
        if matrix:
            _connect_or_set_input_attr(node, input, f'matrixIn[{i}]', is_matrix=True)
        else:
            _connect_or_set_input_attr(node, input, f'input[{i}]')
    
    for target in targets:
        cmds.connectAttr(f'{node}.output', target)
    
    return node


def _create_xyz_input_math_node(node_type: str, inputX: Union[str, float, int]=0, inputY: Union[str, float, int]=0, inputZ: Union[str, float, int]=0, targets: list[str]=None):
    if not isinstance(targets, tuple) and not isinstance(targets, list):
        targets = [targets]
    if targets is None:
        targets = []

    node = cmds.creaeNode(node_type)
    for input, attr in zip((inputX, inputY, inputZ), ('inputX', 'inputY', 'inputZ')):
        _connect_or_set_input_attr(node, input, attr)

    if targets:
        obj, attr = targets[0].split('.')
        target_type = cmds.attributeQuery(attr, node=obj, attributeType=True)
        if target_type == 'double3':
            for target, xyz in zip(targets, 'xyz'):
                cmds.connectAttr(f'{node}.output{xyz}', target)
        else:
            for target in targets:
                cmds.connectAttr(f'{node}.output', target)
            
    return node


def _create_dual_input_math_node(node_type: str, input1: Union[str, float, int], input2: Union[str, float, int], targets: list[str]=None):
    if targets is None:
        targets = []

    node = cmds.createNode(node_type)
    for input, attr in zip((input1, input2), ('input1', 'input2')):
        _connect_or_set_input_attr(node, input, attr)
    
    for target in targets:
        cmds.connectAttr(f'{node}.output', target)

    return node


def _create_single_input_math_node(node_type: str, input: Union[str, float, int, list[int]], targets: list[str]=None, in_matrix: bool=False):
    if targets is None:
        targets = []

    node = cmds.createNode(node_type)
    if isinstance(input, str):
        cmds.connectAttr(input, f'{node}.input')
    elif in_matrix:
        in_attr = 'input' if cmds.objExists(f'{node}.input') else 'inMatrix'
        cmds.setAttr(f'{node}.{in_attr}', input, type='matrix')
    else:
        cmds.setAttr(f'{node}.input', input)
    
    for target in targets:
        output = 'output' if cmds.objExists(f'{node}.output') else 'outMatrix'
        cmds.connectAttr(f'{node}.{output}', target)
    
    return node


def _set_xyz_outputs(node: str, targets: list[str], add_w_output: bool=False):
    if targets is None:
        return
    if isinstance(targets, str):
        targets = [targets]

    obj, attr = targets[0].split('.')
    attr_type = cmds.attributeQuery(attr, node=obj, attributeType=True)
    if attr_type == 'double4' or attr_type == 'double3':
        for target in targets:
            cmds.connectAttr(f'{node}.output', target)
    else:
        xyz = 'XYZW' if add_w_output else 'XYZ'
        for target, axis in zip(targets, xyz):
            cmds.connectAttr(f'{node}.output{axis}', target)


def _connect_or_set_input_attr(node: str, source_attr: Union[str, int, float, list[int]], dest_attr: str, is_matrix: bool=False):
    if isinstance(source_attr, str):
        cmds.connectAttr(source_attr, node + '.' + dest_attr)
    elif is_matrix:
        cmds.setAttr(node + '.' + dest_attr, source_attr, type='matrix')
    else:
        cmds.setAttr(node + '.' + dest_attr, source_attr)

########## Comparison ##########

def create_and_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_dual_input_math_node('and', input, targets)


def create_equal_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_dual_input_math_node('equal', input, targets)


def create_greaterThan_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_dual_input_math_node('greaterThan', input, targets)


def create_lessThan_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_dual_input_math_node('lessThan', input, targets)


def create_max_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('max', input, targets)


def create_min_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('min', input, targets)


def create_not_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('not', input, targets)


def create_or_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_dual_input_math_node('or', input, targets)


########## Operation ##########

def create_absolute_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('max', input, targets)


def create_average_node(inputs: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('average', inputs, targets)


def create_divide_node(inputs: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('divide', inputs, targets)


def create_inverseLerp_node(inputs: list[Union[str, int, float]], targets: list[str]=None, interpolation=0):
    node = _create_dual_input_math_node('inverseLerp', inputs, targets)
    cmds.setAttr(f'{node}.interpolation', interpolation)
    return node


def create_lerp_node(inputs: list[Union[str, int, float]], targets: list[str]=None, weight=0):
    node = _create_dual_input_math_node('lerp', inputs, targets)
    cmds.setAttr(f'{node}.weight', weight)
    return node


def create_log_node(input: Union[str, float, int], targets: list[str]=None, base=2):
    node = _create_single_input_math_node('log', input, targets)
    cmds.setAttr(f'{node}.base', base)
    return node


def create_modulo_node(input: Union[str, float, int], targets: list[str]=None, modulus=1):
    node = _create_single_input_math_node('modulo', input, targets)
    cmds.setAttr(f'{node}.modulus', modulus)
    return node


def create_multiply_node(inputs: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('multiply', inputs, targets)


def create_negate_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('negate', input, targets)


def create_power_node(input: Union[str, float, int], targets: list[str]=None, exponent=2):
    node = _create_single_input_math_node('power', input, targets)
    cmds.setAttr(f'{node}.exponent', exponent)
    return node


def create_subtract_node(inputs: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('subtract', inputs, targets)


def create_sum_node(inputs: list[Union[str, int, float]], targets: list[str]=None):
    return _create_multi_input_math_node('sum', inputs, targets)


########## Rounding ##########

def create_ceil_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('ceil', input, targets)


def create_clampRange_node(input: Union[str, float, int], targets: list[str]=None, minimum=0, maximum=1):
    node = _create_single_input_math_node('clampRange', input, targets)
    cmds.setAttr(f'{node}.minimum', minimum)
    cmds.setAttr(f'{node}.maximum', maximum)
    return node


def create_floor_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('floor', input, targets)


def create_round_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('round', input, targets)


def create_smoothStep_node(input: Union[str, float, int], targets: list[str]=None, leftEdge=0, rightEdge=1):
    node = _create_single_input_math_node('smoothStep', input, targets)
    cmds.setAttr(f'{node}.leftEdge', leftEdge)
    cmds.setAttr(f'{node}.rightEdge', rightEdge)
    return node


def create_truncate_node(input: Union[str, float, int], targets: list[str]=None):
    return _create_single_input_math_node('truncate', input, targets)


########## Matrix ##########

def create_addMatrix_node(input: list[Union[str, int, float]], targets: list[str]=None):
    if targets is None:
        targets = []
    node = _create_multi_input_math_node('addMatrix', input, matrix=True)

    for target in targets:
        cmds.connectAttr(f'{node}.matrixSum', target)

    return node


def create_aimMatrix_node(
    input_matrix: Union[str, list[int]], 
    primary_target_matrix: Union[str, list[int]], 
    secondary_target_matrix: Union[str, list[int]], 
    targets: list[str]=None, 
    primary_input_axis: Union[tuple[str], tuple[Union[int, float]]]=(0, 0, 0), 
    secondary_input_axis: Union[tuple[str], tuple[Union[int, float]]]=(0, 0, 0), 
    primary_mode: int=1, 
    secondary_mode: int=0, 
    primary_target_vector: Union[tuple[str], tuple[Union[int, float]]]=(0, 0, 0), 
    secondary_target_vector: Union[tuple[str], tuple[Union[int, float]]]=(0, 0, 0), 
    pre_space_matrix: Union[str, list[int]]=None, 
    post_space_matrix: Union[str, list[int]]=None
):
    if targets is None:
        targets = []
    node = cmds.createNode('aimMatrix')
    _connect_or_set_input_attr(node, input_matrix, 'inputMatrix', is_matrix=True)
    _connect_or_set_input_attr(node, primary_target_matrix, 'primary.primaryTargetMatrix', is_matrix=True)
    _connect_or_set_input_attr(node, secondary_target_matrix, 'secondary.secondaryTargetMatrix', is_matrix=True)
    for i, axis in enumerate('XYZ'):
        _connect_or_set_input_attr(node, primary_input_axis[i], f'primaryInputAxis{axis}')
        _connect_or_set_input_attr(node, primary_target_vector[i], f'primaryTargetVector{axis}')
        _connect_or_set_input_attr(node, secondary_input_axis[i], f'seocndaryInputAxis{axis}')
        _connect_or_set_input_attr(node, secondary_target_vector[i], f'secondaryTargetVector{axis}')
    cmds.setAttr(f'{node}.primaryMode', primary_mode)
    cmds.setAttr(f'{node}.secondaryMode', secondary_mode)
    _connect_or_set_input_attr(node, pre_space_matrix, 'preSpaceMatrix', is_matrix=True)
    _connect_or_set_input_attr(node, post_space_matrix, 'postSpaceMatrix', is_matrix=True)

    for target in targets:
        cmds.connectAttr(f'{node}.outputMatrix', target)

    return node


def create_axisFromMatrix_node(input: Union[str, list[int]], targets: list[str]=None, axis: int=0):
    node = _create_single_input_math_node(input, in_matrix=True)
    cmds.setAttr(f'{node}.axis', axis)
    _set_xyz_outputs(node, targets)

    return node


def create_blendMatrix_node(input: Union[str, list[int]], target_matrix: Union[str, list[int]], targets: list[str]=None, pre_space_matrix: list[int]=None, post_space_matrix: list[int]=None):
    if not isinstance(target_matrix, list):
        target_matrix = [target_matrix]
    if targets is None:
        targets = []

    node = cmds.createNode('blendMatrix')
    _connect_or_set_input_attr(node, input, 'inputMatrix', is_matrix=True)

    for i, matrix in enumerate(target_matrix):
        _connect_or_set_input_attr(node, matrix, f'target[{i}].targetMatrix', is_matrix=True)

    if pre_space_matrix:
        _connect_or_set_input_attr(node, pre_space_matrix, 'preSpaceMatrix', is_matrix=True)
    if post_space_matrix:
        _connect_or_set_input_attr(node, post_space_matrix, 'postSpaceMatrix', is_matrix=True)

    for target in targets:
        cmds.connectAttr(f'{node}.outputMatrix', target)

    return node


def create_columnFromMatrix_node(in_matrix: Union[str, list[int]], targets: list[str]=None, input=0):
    node = _create_single_input_math_node('columnFromMatrix', in_matrix, in_matrix=True)
    cmds.setAttr(f'{node}.input', input)
    _set_xyz_outputs(node, targets, add_w=True)

    return node


def create_crossProduct_node(input1: list[Union[str, int, float]], input2: list[Union[str, int, float]], targets: str):
    node = cmds.createNode('crossProduct')
    for in_1, in_2, xyz in zip(input1, input2, 'XYZ'):
        _connect_or_set_input_attr(node, in_1, f'input1{xyz}')
        _connect_or_set_input_attr(node, in_2, f'input2{xyz}')

    _set_xyz_outputs(node, targets)

    return node


def create_decomposeMatrix_node(in_matrix: str, targets: list[str]=None, translate: bool=True, rotate: bool=True, scale: bool=True, shear: bool=True):
    """
    Create and connect a decomposeMatrix node.
    """
    if targets is None:
        targets = []
    node = cmds.createNode("decomposeMatrix")
    cmds.connectAttr(in_matrix, f"{node}.inputMatrix")
    for target in targets:
        if translate:
            cmds.connectAttr(f"{node}.outputTranslate", f'{target}.translate')
        if rotate:
            cmds.connectAttr(f"{node}.outputRotate", f'{target}.rotate')
        if scale:
            cmds.connectAttr(f"{node}.outputScale", f'{target}.scale')
        if shear:
            cmds.connectAttr(f"{node}.outputShear",f'{target}.shear')

    return node


def create_determinant_node(input: Union[str, list[int]], targets: list[str]=None):
    return _create_single_input_math_node('determinant', input, targets, in_matrix=True)


def create_dotProduct_node(input1: list[Union[str, int, float]], input2: list[Union[str, int, float]], targets: list[str]):
    node = cmds.createNode('dotProduct')
    for in_1, in_2, xyz in zip(input1, input2, 'XYZ'):
        _connect_or_set_input_attr(node, in_1, f'input1{xyz}')
        _connect_or_set_input_attr(node, in_2, f'input2{xyz}')

    _set_xyz_outputs(node, targets)

    return node


def create_fourByFourMatrix_node(inputs: list[Union[str, int]], targets: list[str]=None):
    if targets is None:
        targets = []
    node = cmds.createNode('fourByFourMatrix')
    input_index = 0
    for i in range(4):
        if input_index == len(inputs):
            break
        for j in range(4):
            if input_index == len(inputs):
                break
            _connect_or_set_input_attr(node, inputs[input_index], f'{i}{j}')
            input_index += 1
    for target in targets:
        cmds.connectAttr(f'{node}.output', target)
    
    return node


def create_holdMatrix_node(input: Union[str, list[int]], targets: list[str]=None):
    return _create_single_input_math_node('holdMatrix', input, targets, in_matrix=True)


def create_inverseMatrix_node(input: Union[str, list[int]], targets: list[str]=None):
    if targets is None:
        targets = []

    node = cmds.createNode('inverseMatrix')
    _connect_or_set_input_attr(node, input, 'inputMatrix')
    for target in targets:
        cmds.connectAttr(f'{node}.outputMatrix', target)

    return node



def create_multiplyPointByMatrix_node(inputs: list[Union[str, int, float]], matrix: Union[str, list[int]], targets: list[str]=None):
    node = _create_single_input_math_node('multiplyPointByMatrix', matrix, in_matrix=True)
    for input, xyz in zip(inputs, 'XYZ'):
        _connect_or_set_input_attr(node, input, f'input{xyz}')

    _set_xyz_outputs(node, targets)

    return node


def create_multiplyVectorByMatrix(inputs: list[Union[str, int, float]], matrix: Union[str, list[int]], targets: list[str]=None):
    if targets is None:
        targets = []
    node = _create_single_input_math_node('multiplyVectorByMatrix', matrix, in_matrix=True)
    for input, xyz in zip(inputs, 'XYZ'):
        _connect_or_set_input_attr(node, input, f'input{xyz}')

    _set_xyz_outputs(node, targets)

    return node


def create_multMatrix_node(in_matrix: list[Union[str, int, float]], targets: list[str]=None):
    node = _create_multi_input_math_node(in_matrix, matrix=True)

    for target in targets:
        cmds.connectAttr(f'{node}.matrixSum', target)

    return node


def create_normalize_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_xyz_input_math_node('normalize', input[0], input[1], input[2], targets)


def create_parentMatrix_node(
    in_matrix: Union[str, list[int]], 
    in_target_matrices: list[Union[str, list[int]]]=None, 
    in_offset_matrices: list[Union[str, list[int]]]=None, 
    in_weights: Union[str, int, float]=None, 
    targets: list[str]=None, 
    pre_space_matrix: Union[str, list[int]]=None, 
    post_space_matrix: Union[str, list[int]]=None
):
    if targets is None:
        targets = []
    node = cmds.createNode('parentMatrix')
    # single input attrs
    for source_attr, dest_attr in zip((in_matrix, 'inputMatrix'), (pre_space_matrix, 'preSpaceMatrix'), (post_space_matrix, 'postSpaceMatrix')):
        _connect_or_set_input_attr(node, source_attr, dest_attr)

    # multi input attrs
    for source_dest_attrs in (in_target_matrices, 'targetMatrix'), (in_offset_matrices, 'offsetMatrix'), (in_weights, 'weight'):
        for source_attr_set, dest_attr in source_dest_attrs:
            if not source_attr_set:
                continue
            for i, source_attr in enumerate(source_attr_set):
                _connect_or_set_input_attr(node, in_matrix, f'target[{i}].{dest_attr}', in_matrix=True)
    
    for target in targets:
        cmds.connectAttr(f'{node}.outputMatrix', target)

    return node


def create_passMatrix_node(input: Union[str, list[int]], targets: list[str]=None, in_scale=2):
    node = _create_single_input_math_node(input, targets, in_matrix=True)
    cmds.setAttr(f'{node}.inScale', in_scale)

    return node


def create_pickMatrix_node(in_matrix: Union[str, list[int]]=None, targets: list[str]=None, scale=True, rotate=True, translate=True, shear=True):
    if targets is None:
        targets = []

    node = cmds.createNode('pickMatrix')
    cmds.setAttr(f'{node}.scale', scale)
    cmds.setAttr(f'{node}.rotate', rotate)
    cmds.setAttr(f'{node}.translate', translate)
    cmds.setAttr(f'{node}.shear', shear)
    if in_matrix:
        _connect_or_set_input_attr(node, in_matrix, 'inputMatrix', in_matrix=True)
    for target in targets:
        cmds.connectAttr(f'{node}.outputMatrix', target)

    return node


def create_pointMatrixMult_node(input: Union[str, list[int]], in_point: list[Union[str, int, float]], targets: list[str]=None, vector_multiply=False):
    node = _create_single_input_math_node('pointMatrixMult', input, targets, in_matrix=True)
    for point, xyz in zip(in_point, 'XYZ'):
        _connect_or_set_input_attr(node, point, f'inPoint{xyz}')
    cmds.setAttr(f'{node}.vectorMultiply', vector_multiply)

    return node


def create_rotationFromMatrix_node(in_matrix: Union[str, list[int]], targets: list[str]=None):
    node = _create_single_input_math_node('rotationFromMatrix', in_matrix, in_matrix=True)
    _set_xyz_outputs(node, targets)

    return node


def create_rowFromMatrix_node(in_matrix: Union[str, list[int]], targets: list[str]=None, input=0):
    node = _create_single_input_math_node('rowFromMatrix', in_matrix, in_matrix=True)
    cmds.setAttr(f'{node}.input', input)
    _set_xyz_outputs(node, targets, add_w=True)

    return node


def create_scaleFromMatrix_node(in_matrix: Union[str, list[int]], targets: list[str]=None):
    node = _create_single_input_math_node('scaleFromMatrix', in_matrix, in_matrix=True)
    _set_xyz_outputs(node, targets)

    return node


def create_translationFromMatrix_node(in_matrix, targets: list[str]=None):
    node = _create_single_input_math_node('translationFromMatrix', in_matrix, in_matrix=True)
    _set_xyz_outputs(node, targets)

    return node

########## Trigonometry ##########

def create_acos_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('acos', input, targets)


def create_asin_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('asin', input, targets)


def create_atan_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('atan', input, targets)


def create_atan2_node(input1: Union[str, int, float], input2: Union[str, int, float], targets: list[str]=None):
    return _create_dual_input_math_node('atan2', input1, input2, targets)


def create_cos_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('cos', input, targets)


def create_sin_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('sin', input, targets)


def create_tan_node(input: Union[str, int, float], targets: list[str]=None):
    return _create_single_input_math_node('tan', input, targets)

########## Other ##########

def create_length_node(input: list[Union[str, int, float]], targets: list[str]=None):
    return _create_xyz_input_math_node('length', input[0], input[1], input[2], targets)


def create_distanceBetween_node(start, end, targets: list[str]=None):
    if targets is None:
        targets = []

    node = cmds.createNode('distanceBetween')
    if isinstance(start, str):
        cmds.connectAttr(start, f'{node}.point1')
    elif isinstance(start, tuple):
        cmds.setAttr(f'{node}.point1', start[0], start[1], start[2])
    else:
        cmds.setAttr(f'{node}.inMatrix1', start, type='matrix')

    if isinstance(end, str):
        cmds.connectAttr(end, f'{node}.point2')
    elif isinstance(end, tuple):
        cmds.setAttr(f'{node}.point2', end[0], end[1], end[2])
    else:
        cmds.setAttr(f'{node}.inMatrix2', end, type='matrix')

    for target in targets:
        cmds.connectAttr(f'{node}.distance', target)
    

    return node

