import math
import random
import statistics
from .accCore import AccCore
from .network import Network
from .layer import ConvLayer, LocalRegionLayer, PoolingLayer, EltwiseLayer, GemmLayer, DPConvLayer
from .util import find_3pair

class LayerTree():
    '''
    layerTree for partition
    property:
    core : architecture parameter of DNN accelerator core
    exe_cyc_dict = the execution cycle list for each sub-layer after partition
    mem_ohd_list = the memory overhead list for each sub-layer after partition
    part_list = the partition list for each layer, (hofm, wofm, nofm)
    utl_list = the utilization list of hardware, (ult_m, ult_n, utl_k)
    '''
    def __init__(self, nn:Network,core:AccCore):
        self.nn = nn
        self.exe_cyc_dict = dict()
        self.part_num = dict()
        self.part_sch_dict = dict()
        self.core = core

    def init_part(self, tot_cores, batch):
        ## initial the exe_cyc, mem_ohd, and partition scheme of every layer
        # the default hardware parallel is Cin-Cout parallel, (h_pe-> Cin/nifm, w_pe-> Cout/nofm),same as TPU, NVDLA.
        # MinMtx: the minimum size of output feature map after partition, (minMtr, minMtr, core.w_pe)
        part_tot = tot_cores // batch 
        if part_tot == 0: part_tot = 1
        for depth in range(self.nn.depth):
            layers_cur = self.nn.layer_idx_bfs[depth]
            layers_list = []
            segments_num = math.ceil(len(layers_cur) / part_tot)
            step = len(layers_cur) // segments_num
            for i in range(segments_num):
                if i != segments_num - 1:
                    layers_list.append(layers_cur[step * i: step *i + step])
                else:
                    layers_list.append(layers_cur[step * i : ])
            exe_cyc_idea = {}
            # print(layers_list)
            for layers in layers_list:
                for lname in layers:
                    layer = self.nn[lname]
                    if isinstance(layer, ConvLayer) or isinstance(layer, GemmLayer):
                        nifm = 1 if isinstance(layer, DPConvLayer) else layer.get_nifm()
                        sfmap = (layer.hofm, layer.wofm, nifm, layer.nofm)
                        mtxu_utl = self.core.cal_mtxu_utl(sfmap)
                        utl = mtxu_utl
                        total_ops = layer.total_ops()
                        exe_cycle = int(total_ops / self.core.h_pe / self.core.w_pe / mtxu_utl[0] / mtxu_utl[1] / mtxu_utl[2])
                    elif isinstance(layer, LocalRegionLayer):
                        sfmap = (layer.hofm, layer.wofm, layer.nifm, layer.nofm)
                        vecu_utl = self.core.vecu_utl(sfmap)
                        utl = vecu_utl
                        total_ops = layer.total_ops()
                        exe_cycle = int(total_ops / self.core.svecu / vecu_utl[0] / vecu_utl[1] / vecu_utl[2])
                        
                    else:
                        raise TypeError('layer type not supported')
                    self.exe_cyc_dict[lname] = exe_cycle
                    exe_cyc_idea[lname] = exe_cycle
                exe_cyc_idea = dict(sorted(exe_cyc_idea.items(), key=lambda item: item[1]))
                exe_cyc_tot = sum(exe_cyc_idea.values())
                part_res = part_tot
                cnt = 0
                for lname, exe_cycle in exe_cyc_idea.items():
                    cnt +=1
                    if cnt < len(exe_cyc_idea):
                        part_num = math.floor(part_tot * (exe_cycle/ exe_cyc_tot))  
                        self.part_num[lname] = part_num if part_num > 0 else 1
                        # print(f'{lname}: {part_num}')
                        part_res = part_res - part_num
                        if part_res == 0:
                            part_res = 1
                            part_num = part_num - 1
                            self.part_num[lname] = part_num
                        elif part_res < 0:
                            raise ValueError(f'In depth {depth}, the number of layer is larger than the number of core. Thus, it can not allocate resource')
                    else:
                        self.part_num[lname] = part_res

    def gen_part_sch(self):
        for lname in self.nn:
            layer = self.nn[lname]
            # print(f'{lname} -> previous {self.nn.prevs(lname)}')
            part_tot = self.part_num[lname]
            hifm = layer.get_hifm()
            wifm = layer.get_wifm()
            nifm = layer.get_nifm()
            hofm = layer.hofm
            wofm = layer.wofm
            nofm = layer.nofm
            # print(f'{lname}, ifm_chw:{nifm, hifm, wifm}, ofm_chw"{nofm, hofm, wofm}, part:{self.part_num[lname]}')
            if isinstance(layer, ConvLayer) or isinstance(layer, GemmLayer):
                ifm_vol = layer.total_ifmap_size()
                wei_vol = layer.total_filter_size()
                hpad = layer.hfil - 1
                wpad = layer.wfil - 1
                nofm = layer.nofm
                wei_vol_perC = wei_vol // nofm
                ofm_vol = layer.total_ofmap_size()
                part_hw_ub = math.ceil((layer.get_hifm() * layer.get_wifm()) / self.core.h_pe)
                part_co_ub = math.ceil((layer.nofm) / self.core.w_pe)
                # print(part_tot, part_hw_ub, part_co_ub)
                if part_co_ub * part_hw_ub <  part_tot:
                    part_hw = part_hw_ub
                    part_co = part_co_ub
                else:
                    if ifm_vol > wei_vol:
                        for i in range(part_hw_ub, 0, -1):
                            if part_tot % i != 0: continue
                            part_hw = i
                            part_co = part_tot // i
                            if part_co_ub < part_co:
                                part_co = part_co_ub
                                part_hw = part_tot // part_co
                            break
                    else:
                        for i in range(part_co_ub, 0, -1):
                            if part_tot % i != 0: continue
                            part_co = i
                            part_hw = part_tot // i
                            if part_hw > part_hw_ub:
                                part_hw = part_hw_ub
                                part_co = part_tot // part_hw
                            break

            elif isinstance(layer, LocalRegionLayer):
                part_hw_ub = layer.get_hifm() * layer.get_wifm()
                part_co_ub = math.ceil(layer.nofm / self.core.svecu)
                ifm_vol = layer.total_ifmap_size()
                ofm_vol = layer.total_ofmap_size()
                wei_vol_perC = 0
                hpad = layer.hreg - 1
                wpad = layer.wreg - 1
                nofm = layer.nofm 
                wei_vol = wei_vol_perC * nofm
                if part_co_ub * part_hw_ub <  part_tot:
                    part_hw = part_hw_ub
                    part_co = part_co_ub
                else:
                    for i in range(part_co_ub, 0, -1):
                        if part_tot % i != 0: continue
                        part_co = i
                        part_hw = part_tot // i
                        if part_hw > part_hw_ub:
                            part_hw = part_hw_ub
                            part_co = part_tot // part_hw
            else:
                raise TypeError(f'layer type {layer} not supported')
            part_ho = round(math.sqrt(part_hw))
            part_wo = part_hw // part_ho
            isDpconv = isinstance(layer, DPConvLayer)
            # print(f'In partition stage, task of {lname} will be part {[part_ho, part_wo, part_co]}.')
            new_sch, task_vol = self.check_ubuf_validate([part_ho, part_wo, part_co], hifm, wifm, nifm, wei_vol_perC, hofm, wofm, nofm, hpad, wpad, isDpconv)
            # print(f'In partition stage, task of {lname} will be part {new_sch} and will occpuy {task_vol} Bytes.')
            # print(ifm_vol, wei_vol, ofm_vol)
            # print(f'info:{[hifm, wifm, nifm, wei_vol_perC, hofm, wofm, nofm]}')
            self.part_sch_dict[lname] = new_sch

    
    def check_ubuf_validate(self, part_sch, hifm, wifm, nifm, wei_vol_per,hofm, wofm, nofm, hpad, wpad, isDpconv):
        ubuf_size  =self.core.get_buf_size('ubuf')
        part_h, part_w, part_co = part_sch[0], part_sch[1], part_sch[2]
        hifm_part = math.ceil(hifm/ part_h) + hpad
        wifm_part = math.ceil(wifm/ part_w) + wpad
        nifm_part = math.ceil(nifm/ part_co) if isDpconv else nifm
        nofm_part = math.ceil(nofm / part_co)
        hofm_part = math.ceil(hofm /part_h)
        wofm_part = math.ceil(wofm/part_w)
        ifm_vol = hifm_part * wifm_part * nifm_part
        wei_vol = wei_vol_per * nofm_part
        ofm_vol = nofm_part * hofm_part * wofm_part
        task_vol = ifm_vol + wei_vol + ofm_vol
        while task_vol > ubuf_size:
            if ifm_vol > wei_vol and (hifm_part > 1 or wifm_part > 1):
                if part_h > part_w and hifm_part > 1:
                    part_w +=1
                else:
                    part_h +=1
            else:
                part_co += 1
            hifm_part = math.ceil(hifm/ part_h)+ hpad
            wifm_part = math.ceil(wifm/ part_w)+ wpad
            nifm_part = math.ceil(nifm/ part_co) if isDpconv else nifm
            ifm_vol = hifm_part * wifm_part * nifm_part
            nofm_part = math.ceil(nofm / part_co)
            hofm_part = math.ceil(hofm /part_h)
            wofm_part = math.ceil(wofm/part_w)
            wei_vol = wei_vol_per * nofm_part
            ofm_vol = nofm_part * hofm_part * wofm_part
            task_vol = ifm_vol + wei_vol + ofm_vol
        return [part_h, part_w, part_co], task_vol
    
    def clear(self):
        self.exe_cyc_dict.clear()
        self.part_num.clear()
        self.part_sch_dict.clear()
    
    def __str__(self):
        str_ = 'layer name:\ttotal part number\tpart_sch [h, w, c]\n'
        for lname, part_sch in self.part_sch_dict.items():
            part_tot = self.part_num[lname]
            str_ += f'{lname}:\t{part_tot}\t{part_sch}\n'
        return str_
        


