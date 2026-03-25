import math
import numpy as np
import subprocess
from core import Layer, ConvLayer, LocalRegionLayer, GemmLayer

## find all the possible multiplication pair for give positive int result
def find_2pair(data):
    '''
    :param data: result of multiplication, data = a * b
    :return: list of all pairs [(a, b),...]
    '''
    pair_list = []
    for i in range(1,data+1):
        if data % i == 0:
            pair_list.append((i,data//i))
    return pair_list

def find_3pair(data):
    '''
    :param data: result of multiplication, data = a * b * c
    :return: list of all pairs [(a, b, c),...]
    '''
    pair_list = []
    pair2_list = find_2pair(data)
    for abc in pair2_list:
        ab = abc[0]
        pair_ab_list = find_2pair(ab)
        for pair_ab in pair_ab_list:
            pair_list.append(pair_ab + (abc[1],))
    return pair_list

def get_part_range(size, part_len):
    '''
    get the range list from the part_len and the size of h/w/nofm
    :param size:
    :param part_len:
    :return: a range list [0, step, ... n_step, size]
    '''
    p_range = []
    start = size
    step = math.ceil(size / part_len)
    # for i in range(part_len):
    #     if step <= start:
    #        p_range.append(step * i)
    #     else:
    #         p_range.append(step * i + start - step)
    #         if start < 0:
    #             break
    #     start = start - step
    i = 0
    while i * step < size:
        p_range.append(i * step)
        i += 1
    p_range.append(size)
    # assert len(p_range) == part_len + 1
    return p_range

def get_part_range_limit(size,part_len, minstep):
    '''
    get the range list from the part_len and the size of h/w/nofm
    in partition, the partition can make sure the part_len unchanged
    :param size:
    :param part_len:
    :return: a range list [0, step, ... n_step, size]
    '''
    p_range = [0]
    start = size
    step = max(math.ceil(size / part_len), minstep)
    for i in range(1,part_len):
        if step <= start:
            p_range.append(step * i)
        else:
            p_range.append(start)
        start = start - step
        if start <= step:
            break
    p_range.append(size)
    # assert len(p_range) == part_len + 1
    return p_range

class Range():
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def length(self):
        return self.end - self.start

    def dependent(self, other):
        return (self.start <= other.start < self.end) or (other.start <= self.start and self.start < other.end)

    def isSame(self, other):
        return self.start == other.start and self.end == other.end

    def __str__(self):
        str_ = '[ ' + str(self.start) + ', ' +str(self.end) + ')'
        return str_
    
class FmapRange():
    def __init__(self, bRange:Range, hRange:Range, wRange:Range, cRange:Range):
        self.bRange = bRange
        self.hRange = hRange
        self.wRange = wRange
        self.cRange = cRange

    def volume(self):
        return  self.bRange.length() * self.hRange.length() * self.wRange.length() * self.cRange.length()

    def dependent(self, other):
        return self.bRange.isSame(other.bRange) and self.hRange.dependent(other.hRange) and self.wRange.dependent(other.wRange) and self.cRange.dependent(other.cRange)

    def overlap_fmap(self, other):
        bs = max(self.bRange.start, other.bRange.start)
        be = min(self.bRange.end, other.bRange.end)
        hs = max(self.hRange.start, other.hRange.start)
        he = min(self.hRange.end, other.hRange.end)
        ws = max(self.wRange.start, other.wRange.start)
        we = min(self.wRange.end, other.wRange.end)
        cs = max(self.cRange.start, other.cRange.start)
        ce = min(self.cRange.end, other.cRange.end)
        if not ( bs < be and hs < he and ws < we and cs < ce):
            print(bs, be, hs, he, ws, we, cs)
        assert bs < be, (f'b start:{bs}, b end :{be}')
        assert hs < he, (f'h start:{hs}, h end :{he}')
        assert ws < we, (f'w start:{ws}, w end :{we}')
        assert cs < ce, (f'cs start:{cs}, cs end :{ce}')
        brange = Range(bs, be)
        hrange = Range(hs, he)
        wrange = Range(ws, we)
        crange = Range(cs, ce)
        return FmapRange(brange, hrange, wrange, crange)

    def overlap_volume(self, other):
        overlap_fmap = self.overlap_fmap(other)
        return overlap_fmap.volume()

    def __str__(self):
        str_ = '[b,  h, w, c]: ' + '[' + str(self.bRange) + str(self.hRange) + ', ' + str(self.wRange) + ', ' + str(self.cRange) + ' ]'
        return str_

def ofm_to_ifm(layer:Layer, hRange:Range, wRange:Range):
    padding_rate = 0
    if isinstance(layer, ConvLayer):
        zeroPad = (layer.hfil -1) // 2
        h_start_zp = hRange.start * layer.htrd -zeroPad
        h_end_zp = h_start_zp + (hRange.length() - 1) * layer.htrd + (layer.hfil - 1) +1
        ## exclude the zero padding
        h_start = max(h_start_zp, 0)
        h_end = min(h_end_zp, layer.hifm - layer.hfil + 1)

        zeroPad = (layer.wfil -1) // 2
        w_start_zp = wRange.start * layer.wtrd - zeroPad
        w_end_zp = w_start_zp + (wRange.length() - 1) * layer.wtrd + (layer.wfil - 1) + 1
        ## exclude the zero padding
        w_start = max(w_start_zp, 0)
        w_end = min(w_end_zp, layer.wifm - layer.hfil + 1)
        padding_rate = 1 - ((h_end - h_start) * (w_end - w_start) / ((h_end_zp - h_start_zp) * (w_end_zp - w_start_zp)))
    
    elif isinstance(layer, GemmLayer):
        h_start = hRange.start
        h_end = h_start + hRange.length()
        w_start = wRange.start
        w_end = w_start + wRange.length() 

    elif isinstance(layer,LocalRegionLayer):
        h_start = hRange.start * layer.htrd if layer.up == 0 else hRange.start // layer.htrd
        h_end = h_start + hRange.length() * layer.htrd if layer.up == 0 else h_start + hRange.length() // layer.htrd
        w_start = wRange.start * layer.wtrd if layer.up == 0 else wRange.start // layer.wtrd
        w_end = w_start + wRange.length() * layer.wtrd if layer.up==0 else w_start + wRange.length() // layer.wtrd

    else:
        raise ValueError('layer type not supported')
    return Range(h_start, h_end), Range(w_start, w_end), padding_rate

def idx2coreidx(idx, mask:np.array):
    '''
    mask: the allocated core for a network workload, True: used, False: not used 
    '''
    # xlen = shape[0]
    # ylen = shape[1]
    # y = idx // xlen
    # x = idx % xlen
    # x, y = 0, 0
    xlen = mask.shape[0]
    ylen = mask.shape[1]
    cnt = 0
    for j in range(0, ylen):
        for i in range(0,xlen):
            if mask[i][ylen-j-1]:
                cnt = cnt + 1
                if cnt == idx + 1:
                    return (i,j)
    raise ValueError(f'{idx} not in mask {mask}')

def coreidx2idx(coreidx, mask):
    xlen = mask.shape[0]
    ylen = mask.shape[1]
    cnt = 0
    for j in range(0, ylen):
        for i in range(0, xlen):
            if mask[i][ylen-j-1]:
                if coreidx[0] == i and coreidx[1] == ylen-j-1:
                    return cnt
                cnt = cnt +1
    raise ValueError(f'{coreidx} not in mask {mask}')

# def coreidx2idx(coreidx, shape):
#     xlen = shape[0]
#     ylen = shape[1]
#     return coreidx[0] + coreidx[1] * xlen

def find_center_coreidx(coreidx_list):
    x_tot = 0
    y_tot = 0
    for coreidx in coreidx_list:
        x_tot += coreidx[0]
        y_tot += coreidx[1]
    center_idx_tmp = (x_tot // len(coreidx_list), y_tot // len(coreidx_list))
    dist = 1000
    for coreidx in coreidx_list:
        dist_tmp = manhattan_distance(center_idx_tmp, coreidx)
        if dist_tmp < dist:
            dist = dist_tmp
            center_idx = coreidx
    return center_idx 

def find_coreidx_dist(c_coreidx, dist, core_vld_mask, shape):
    x_c = c_coreidx[0]
    y_c = c_coreidx[1]
    xlen = shape[0]
    ylen = shape[1]
    coreidx_list = []
    for x_dist in range(dist):
        y_dist = dist - x_dist
        x_tmp = x_c + x_dist
        y_tmp = y_c + y_dist
        if x_tmp < xlen and y_tmp < ylen:  # available coreidx
            idx_tmp = coreidx2idx((x_tmp, y_tmp), shape)
            if core_vld_mask[idx_tmp] == 1:
                coreidx_list.append(idx_tmp)
    return coreidx_list

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def eu_distance_sqrt(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# def find_optimal_mapping(intra_reuse, core_vld_mask):
def gen_map_order(mesh_shape, center_idx):
    '''
    mesh_shape: (xlen, ylen)
    center_idx: based on center core idx (x,y), generate a list based on the mahattan distance
    '''
    xlen = mesh_shape[0]    
    ylen = mesh_shape[1]
    x = center_idx[0]
    y = center_idx[1]
    assert 0<= x < xlen and 0<= y < ylen
    core_map_prior = [[(x, y)]]
    for mhd in range(1, (xlen - 1) + (ylen -1) + 1):
        core_tmp = []
        for mhd_x in range(mhd+1):
            mhd_y = mhd - mhd_x
            x_l = x - mhd_x
            x_r = x + mhd_x
            y_d = y - mhd_y
            y_t = y + mhd_y
            if (x_l,y_d) not in core_tmp and 0<= x_l < xlen and 0<= y_d < ylen:
                core_tmp.append((x_l,y_d))
            if (x_l,y_t) not in core_tmp and 0<= x_l < xlen and 0<= y_t < ylen:
                core_tmp.append((x_l,y_t))
            if (x_r,y_d) not in core_tmp and 0<= x_r < xlen and 0<= y_d < ylen:
                core_tmp.append((x_r,y_d))
            if (x_r,y_t) not in core_tmp and 0<= x_r < xlen and 0<= y_t < ylen:
                core_tmp.append((x_r,y_t))
        if len(core_tmp) == 0: break
        core_map_prior.append(core_tmp)
    return core_map_prior


def run_command(command:list):
    '''
    call the the linux terminal command:
    command: e.g. ls -l in command list is ['ls','-l']
    '''
    try:
        # 调用外部的 shell 脚本
        # print(command)
        result = subprocess.run(
            command,  # bash
            check=True,             # if not return 0, call error
            text=True,              # return string not Byte
            capture_output=True      
        )
        # print("output:")
        # print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("error:", e)
        print("std error:")
        print(e.stderr)

def find_hotpoint(file,unitK=True):
    grid_file = open(file, 'r')
    content = grid_file.readlines()
    max_temp = 0
    max_temp_layer = 0
    for field in content:
        str_tmp = field.split()
        name = str_tmp[0]
        if 'Layer' in name:
            layer_num = int(str_tmp[1][0:-1])
        else:
            grid_temp = float(str_tmp[1][0:-1])
            if max_temp < grid_temp:
                max_temp = grid_temp
                max_temp_layer = layer_num
    unit = 'k'
    if unitK is False:
        max_temp = max_temp - 273.15
        unit = 'C'
    # print(f'The peak tempurature is {max_temp}{unit}, allocated at layer {max_temp_layer}')
    grid_file.close()
    return max_temp


        



