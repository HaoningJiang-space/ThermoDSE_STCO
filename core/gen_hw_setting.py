from .tsmc28_lib import *
from .tech_params import TechParams
import math

def mtxu_settting_gen(smtxu, area_scale=1, energy_scale=1, lib_type='28nm'):
    if lib_type == '28nm':
        pe_model_unit = mac_int8()
    else:
        raise ValueError('the lib type is not supported.')
    pe_num  = smtxu[0] * smtxu[1]
    area = pe_num  * pe_model_unit.area / area_scale
    return area, pe_model_unit.cost / energy_scale

def vecu_settting_gen(svecu, area_scale=1, energy_scale=1, lib_type='28nm'):
    if lib_type == '28nm':
        pe_model_unit = mac_fp16()
    else:
        raise ValueError('the lib type is not supported.')
    pe_num = svecu
    area = pe_num * pe_model_unit.area / area_scale
    return area, pe_model_unit.cost / energy_scale

def sram_setting_gen(capacity, area_scale=1, energy_scale=1, lib_type='28nm'):
    if lib_type == '28nm':
        if capacity > 128*1024:
            sram_model_unit = sram_128KB()
            area_enlarge = capacity / (128*1024)
            energy_enlarge = (capacity / (128*1024))**0.6
        elif capacity > 64*1024:
            sram_model_unit = sram_64KB()
            area_enlarge = capacity / (64*1024)
            energy_enlarge = capacity / (64*1024)
        else:
            sram_model_unit = sram_32KB()
            area_enlarge = capacity / (32*1024)
            energy_enlarge = area_enlarge
    else:
        raise ValueError('the lib type is not supported.')
    area = sram_model_unit.area * area_enlarge / area_scale
    return area, sram_model_unit.rcost * energy_enlarge / energy_scale, sram_model_unit.wcost * energy_enlarge / energy_scale

def regf_setting_gen(capacity, area_scale=1, energy_scale=1, lib_type='28nm'):
    if lib_type == '28nm':
        regf_model_unit = regf_1KB()
        scale = capacity / (1*1024)
    else:
        raise ValueError('the lib type is not supported.')
    area = regf_model_unit.area * scale / area_scale
    return area, regf_model_unit.rcost/energy_scale, regf_model_unit.wcost/energy_scale

def nop_setting_gen(ics, nop_bw, tech: TechParams = None):
    ## D2D interconnect technology -- see TechParams.d2d for defaults/references.
    ## The ics range is [0.5, 3.5] mm
    if tech is None:
        tech = TechParams()
    if ics < 0.5 or ics > 3.5:
        raise ValueError(f'Inter-chiplet Space ranges from 0.5mm to 5mm, the input ICS is {ics} mm')
    nop_cost = tech.d2d.energy_per_bit_pJ
    nop_area = tech.d2d.pin_area_um2 * nop_bw / tech.d2d.bandwidth_density_GBps_per_pin
    return nop_area, nop_cost

def yield_setting_gen(area_per_chiplet, tech: TechParams = None):
    # area_per_chiplet: unit: m^2
    ## yield model: Y = (1 + A * D0/alpha) ^ (-alpha) -- see TechParams.yield_model for defaults/references.
    if tech is None:
        tech = TechParams()
    D0 = tech.yield_model.defect_density_per_cm2
    alpha = tech.yield_model.cluster_alpha
    return (1 + area_per_chiplet * 10**4 * D0 / alpha ) ** (-alpha)