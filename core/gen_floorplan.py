import os
import math
from .util import run_command


class block_flp:
      def __init__(self,name,width,height,leftX,bottomY):
            self.name = name
            self.width = width
            self.height = height
            self.leftX = leftX
            self.bottomY = bottomY

      def covert(self, chip_ics, xIdx, yIdx, coreWidth, coreHeight, sizeTile_x):
            leftX = round(xIdx * (coreWidth + chip_ics) + self.leftX, 6)
            bottomY = round(yIdx * (coreHeight + chip_ics) + self.bottomY, 6)
            outstring = self.name + "_" + str(yIdx*sizeTile_x + xIdx) + "\t"+ str(self.width) +"\t" +str(self.height) + "\t" + str(leftX) + "\t" + str(bottomY) + "\n"
            return outstring      

      def covert_chiplet(self, chip_ics,NoC_space, xIdx, yIdx, coreWidth, coreHeight, sizeTile_x, xchipIdx, ychipIdx, edge_w, edge_h):
            leftX = round(edge_w + xIdx * coreWidth + NoC_space * (xIdx - xchipIdx) + xchipIdx * chip_ics + self.leftX, 6)
            bottomY = round(edge_h + yIdx * coreHeight + NoC_space * (yIdx - ychipIdx) + ychipIdx * chip_ics + self.bottomY, 6)
            outstring = self.name + "_" + str(yIdx*sizeTile_x + xIdx) + "\t"+ str(self.width) +"\t" +str(self.height) + "\t" + str(leftX) + "\t" + str(bottomY) + "\n"
            return outstring

class floorplan_generator():
    '''
        call hotfloorplan to generate the floorplan based on the hardware setting.
        hotfloorplan needs 3 files. 1. config file (use example), 2. .desc (describe the area and connectivty) 3. avg.p (average power of each module)
        parameter
        stack type: 2D, 2.5D, 3D
        design info: a dict (key: module name, value:[area, avg_power])
    '''
    def __init__(self,spec_info:dict, connectivity:list, path = '../tmp', hs_path ='../../HotSpot',  stack_type='2D') -> None:
        self.stack_type = stack_type
        self.spec_info = spec_info 
        self.connectivity = connectivity
        self.path = path
        self.hs_path = hs_path
    
    def gen_floorplan(self,chip_ics, xx, yy):
        '''
        generate the floorplan of ONE accelerator core first and scale it according to the ics, xx, and yy.
        this is an important function to generate the floorplan of chiplet design for multi-core Accelerataors
        chip_ics: inter-chiplet space, adjusting this may can affect on the temperature
        xx, yy  : the shape of chiplet core e.g. 2x2, 2x3, 4x4
        '''
        flp_path = os.path.join(self.path,'floorplan')
        os.makedirs(os.path.dirname(flp_path), exist_ok=True)
        desc_fname = 'core.desc'
        avgp_fname = 'avg.p'
        config_fname = 'example.config'
        desc_path = os.path.join(flp_path,desc_fname)
        avgp_path = os.path.join(flp_path,avgp_fname)
        hotfloorplan = os.path.join(self.hs_path,'hotfloorplan')
        config_path = os.path.join(flp_path,config_fname)
        desc_file = open(desc_path, 'w')
        avgp_file = open(avgp_path, 'w')
        header_line = '# Area and aspect ratios of blocks\n# Line Format: <unit-name>\t<area-in-m2>\t<min-aspect-ratio>\t<max-aspect-ratio>\t<rotable>\n'
        desc_file.write(header_line)
        for module, info in self.spec_info.items():
            area = info[0]
            avgp = info[1]
            if 'mtxu' in module:
                desc_tmp = '{}\t{}\t1\t1\t1\n'.format(module,area)
            else:
                 desc_tmp = '{}\t{}\t1\t3\t1\n'.format(module,area)
            avgp_tmp = '{}\t{}\n'.format(module,avgp)
            desc_file.write(desc_tmp)
            avgp_file.write(avgp_tmp)
        avgp_file.close()
        header_line = '\n# Connectivity information\n# Line format <unit1-name>\t<unit2-name>\t<wire_density>\n'
        desc_file.write(header_line)
        for connect_info in self.connectivity:
            desc_file.write(connect_info+'\n')
        desc_file.close()
        single_flp_path = os.path.join(flp_path,'single_core.flp')
        command = [hotfloorplan, '-c', config_path, '-f', desc_path, '-p', avgp_path, '-o', single_flp_path]
        run_command(command)
        output_file = os.path.join(flp_path,'output.flp')
        print(xx, yy)
        coreWidth, coreHeight = gen_tile_flp(single_flp_path,output_file,chip_ics,xx, yy)
        output_file = os.path.join(flp_path,'interposer.flp')
        gen_interposer_flp(coreWidth, coreHeight, output_file)
        output_file = os.path.join(flp_path,'interposer_TIM.flp')
        gen_interposer_flp(coreWidth, coreHeight, output_file)


    def run_hotspot(self,nn_name):
         config_file = os.path.join(self.path,'example.config')
         flp_file = os.path.join(self.path,'floorplan','output.flp')
         ptrace_file = os.path.join(self.path, 'ptrace', nn_name + '.ptrace')
         shell = os.path.join(self.path,'run.sh')
         command = [shell, config_file, flp_file, ptrace_file]
         run_command(command)

def fixed_floorplen_gen(spec_info:dict, flp_file = '../tmp/floorplan/single_core_3D.flp'):
    mtxu_area = spec_info['mtxu'][0]
    mtxu_h_w = round(math.sqrt(mtxu_area),4) #fix the aspect ratio of mtxu is 1
    ubuf_area = spec_info['ubuf'][0]
    ibuf_area = spec_info['l0a'][0] + spec_info['l0b'][0] 
    obuf_area = spec_info['l0c'][0] + spec_info['l1c'][0]
    vecu_area = spec_info['vecu'][0]
    io_area =  0.5 * (mtxu_area + ubuf_area + ibuf_area + obuf_area + vecu_area)
    ibuf_h = round(ibuf_area/mtxu_h_w, 5)
    vecu_h = round(obuf_area/mtxu_h_w, 5)
    obuf_h = round(vecu_area/mtxu_h_w, 5)
    core_h = round(mtxu_h_w + ibuf_h + vecu_h + obuf_h, 5)
    ubuf_w = round(ubuf_area/core_h, 5)
    if ubuf_w == 0: ubuf_w = 0.00001
    if ibuf_h == 0: ibuf_h = 0.00001
    if vecu_h == 0: vecu_h = 0.00001
    if obuf_h == 0: obuf_h = 0.00001
    core_w = mtxu_h_w + ubuf_w
    io_width = round((math.sqrt(((core_h + core_w)/2)**2+ io_area) - (core_w+core_h) /2)/2, 5)
    if io_width == 0: io_w = 0.00001
    # print(f'insert_w: {insert_w}, core_h:{core_h}, core_w:{ubuf_w + mtxu_h_w}')
    # insert_w = (core_h > ubuf_w + mtxu_h_w)
    # if insert_w:  # insert io at w dimension
    #      io_w = round(core_h - ubuf_w - mtxu_h_w , 5)
    #      if io_w == 0: io_w = 0.00001
    #      core_w = ubuf_w + mtxu_h_w + io_w
    #      core_h = core_h
    #      io_h = core_h
    # else:        # insert io at h dimension
    #     io_h = round(ubuf_w + mtxu_h_w - core_h , 5)
    #     if io_h == 0: io_h = 0.00001
    #     core_w = ubuf_w + mtxu_h_w 
    #     core_h = core_h + io_h
    #     io_w = core_w

    flp_writer = open(flp_file,'w')
    # # flp file Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]
    # str_  = 'obuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, obuf_h, 0.000000, 0.000000)
    # str_ += 'vecu\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, vecu_h, 0.000000, obuf_h)
    # str_ += 'mtxu\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, mtxu_h_w, 0.000000, obuf_h + vecu_h)
    # str_ += 'ibuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, ibuf_h, 0.000000, obuf_h + vecu_h + mtxu_h_w)
    # str_ += 'ubuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(ubuf_w, core_h, mtxu_h_w, 0.000000)
    # if insert_w:
    #     str_ += 'io\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n\n'.format(io_w, io_h, mtxu_h_w + ubuf_w, 0.000000)
    # else:
    #     str_ += 'io\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n\n'.format(io_w, io_h,0.000000, mtxu_h_w + ibuf_h + vecu_h + obuf_h)
    str_  = 'obuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, obuf_h, io_width, io_width) 
    str_ += 'vecu\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, vecu_h, io_width, io_width + obuf_h)
    str_ += 'mtxu\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, mtxu_h_w, io_width, io_width + obuf_h + vecu_h)
    str_ += 'ibuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(mtxu_h_w, ibuf_h, io_width, io_width + obuf_h + vecu_h + mtxu_h_w)
    str_ += 'ubuf\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(ubuf_w, core_h, io_width + mtxu_h_w, io_width)
    str_ += 'io_0\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(io_width, core_h + io_width, 0.000000, 0.000000)
    str_ += 'io_1\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(core_w + io_width, io_width, io_width, 0.000000)
    str_ += 'io_2\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(io_width, core_h + io_width, core_w + io_width, io_width)
    str_ += 'io_3\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(core_w + io_width, io_width, 0.000000, core_h + io_width)
    flp_writer.write(str_)
    flp_writer.close()
    # print(core_w, core_h)
    return core_w + io_width * 2, core_h + io_width * 2

## inF: input floorplan file of single core
def gen_tile_flp(inF, outF, chip_ics, xx, yy):
    '''
    this is an important function to generate the floorplan of chiplet design for multi-core Accelerataors
    shape_chip: the shape of chiplet core e.g. 2x2, 2x3, 4x4
    chip_ics: inter-chiplet space, adjusting this may can affect on the temperature
    '''
    with open(inF, 'r') as infile:
            lines = infile.readlines()
    blocks = []
    coreX = 0
    coreY = 0
    for line in lines:
        #   print(line)
          fields = line.strip().split('\t')
          if "#" not in line and len(fields) > 1:
                name = fields[0]
                # print(fields)
                width = round(float(fields[1]),6) 
                height = round(float(fields[2]),6)
                leftX = round(float(fields[3]), 6)
                bottomY = round(float(fields[4]), 6)
                ## if the name is constructed with first '_'，the created dead block by hotfloorplan
                if name[0] != '_':  
                    block_tmp = block_flp(name,width,height,leftX,bottomY)
                    blocks.append(block_tmp)
                coreX = max(leftX + width,coreX)
                coreY = max(bottomY + height,coreY)
#     core_intvl = max(coreY, coreX)
    outFile = open(outF, 'w')
    print("core width:" + str(coreX)+ ", core height:" + str(coreY))
    for i in range(yy):
        for j in range(xx):
            for n in range(len(blocks)):
                block_tmp = blocks[n]
                yIdx = i
                xIdx = j
                string_tmp = block_tmp.covert(chip_ics,xIdx,yIdx,coreX,coreY,xx)
                outFile.write(string_tmp)
    outFile.write('\n')
    outFile.close()
    tile_width = round((coreX + chip_ics) * xx - chip_ics, 6) 
    tile_height = round((coreY + chip_ics) * yy - chip_ics, 6)
    return tile_width, tile_height

def gen_cover_flp(name, coreWidth, coreHeight, eblk_w, eblk_h, outF='interposer.flp'):
    outFile = open(outF, 'w')
    str_  = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name+ '_e0', eblk_w + coreWidth, eblk_h , 0, 0)
    str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name+ '_e1', eblk_w, eblk_h + coreHeight, 0, eblk_h)
    str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name+ '_e2', eblk_w + coreWidth, eblk_h , eblk_w, eblk_h + coreHeight)
    str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name+ '_e3', eblk_w, eblk_h + coreHeight, eblk_w + coreWidth, 0)
    # write_str = '\n'+ name +'\t' + str(coreWidth) + '\t' + str(coreHeight) + '\t' + '0.000000\t' + '0.000000\t\n\n'
    str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, coreWidth,coreHeight, eblk_w, eblk_h)
    outFile.write(str_)
    outFile.close()


# def gen_chip_fp(self, shape_chip, chip_ics, path='../tmp'):
#     '''
#     this is an important function to generate the floorplan of chiplet design for multi-core Accelerataors
#     shape_chip: the shape of chiplet core e.g. 2x2, 2x3, 4x4
#     chip_ics: inter-chiplet space, adjusting this may can affect on the temperature
#     '''
#     xlen = shape_chip[0]
#     ylen = shape_chip[1]

class floorplan_generator_3D():
    def __init__(self, spec_info, path = '../tmp', hs_path ='../../HotSpot', stack_type='2.5D'):
        self.stack_type = stack_type
        self.spec_info = spec_info 
        self.path = path
        self.hs_path = hs_path
        self.core_w = 0
        self.core_h = 0
        self.ics = 0

    
    def gen_core_floorplan(self):
        flp_path = os.path.join(self.path,'floorplan','single_core_3D.flp')
        self.core_w, self.core_h = fixed_floorplen_gen(self.spec_info, flp_path) 
    
    def gen_sys_floorplan(self,w_const, h_const, ics, xx, yy, xcut, ycut, NoC_space = 0.0002, sys_flp_name = 'output_3D.flp'):
        core_flp_path = os.path.join(self.path,'floorplan','single_core_3D.flp')
        sys_flp_path = os.path.join(self.path,'floorplan', sys_flp_name)
        with open(core_flp_path, 'r') as infile:
            lines = infile.readlines()
        blocks = []
        coreX = 0
        coreY = 0
        for line in lines:
            #   print(line)
            fields = line.strip().split('\t')
            if "#" not in line and len(fields) > 1:
                    name = fields[0]
                    # print(fields)
                    width = round(float(fields[1]),5) 
                    height = round(float(fields[2]),5)
                    leftX = round(float(fields[3]), 5)
                    bottomY = round(float(fields[4]), 5)
                    ## if the name is constructed with first '_'，the created dead block by hotfloorplan
                    if name[0] != '_':  
                        block_tmp = block_flp(name,width,height,leftX,bottomY)
                        blocks.append(block_tmp)
                    coreX = max(leftX + width,coreX)
                    coreY = max(bottomY + height,coreY)
        # generate the entire floorplan with chiplet system
        outFile = open(sys_flp_path,'w')
        self.ics = ics
        xcore = xx//xcut
        ycore = yy//ycut
        die_h = self.core_h * ycore + NoC_space * (ycore - 1)
        die_w = self.core_w * xcore + NoC_space * (xcore - 1)
        sys_width  = (die_w + self.ics) * xcut - self.ics
        sys_height = (die_h + self.ics) * ycut - self.ics
        # print(self.core_h, self.core_w , die_h, die_w, sys_height, sys_width)
        # print(die_h, die_w, sys_height, sys_width)
        # assert sys_height <= h_const, sys_width <= w_const
        if sys_width < w_const or sys_height < h_const:
            eblk_w = (w_const - sys_width) / 2
            eblk_h = (h_const - sys_height) /2
        else:
            eblk_w = 0.001
            eblk_h = 0.001
        str_  = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk0', eblk_w + sys_width, eblk_h , 0, 0)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk1', eblk_w, eblk_h + sys_height, 0, eblk_h)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk2', eblk_w + sys_width, eblk_h , eblk_w, eblk_h + sys_height)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk3', eblk_w, eblk_h + sys_height, eblk_w + sys_width, 0)
        outFile.write(str_)
        for i in range(ycut):
            for j in range(xcut):
                for ii in range(ycore):
                    for jj in range(xcore):
                        for n in range(len(blocks)):
                            block_tmp = blocks[n]
                            xchipIdx, ychipIdx = j, i
                            yIdx = ychipIdx * ycore + ii
                            xIdx = xchipIdx * xcore + jj
                            string_tmp = block_tmp.covert_chiplet(self.ics, NoC_space,xIdx, yIdx, self.core_w, self.core_h, xx, xchipIdx, ychipIdx, eblk_w, eblk_h)
                            outFile.write(string_tmp)
                        
                        idx = yIdx * xx + xIdx
                        if xIdx < xx -1:
                            name = 'blockX_{}'.format(idx)
                            width = self.ics if (xIdx + 1) % xcore == 0 else NoC_space
                            height = self.core_h
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics + self.core_w
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics
                            # str_ = name + '\t' + str(width) + '\t' + str(height) + '\t' + str(leftX) + '\t' + str(bottomY) + '\n'
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
                        if yIdx < yy - 1:
                            name = 'blockY_{}'.format(idx)
                            width = self.core_w
                            height = self.ics if (yIdx + 1) % ycore == 0 else NoC_space
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics      
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics + self.core_h
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
                        if xIdx < xx - 1 and yIdx < yy - 1:
                            name = 'blockXY_{}'.format(idx)
                            width = self.ics if (xIdx + 1) % xcore == 0 else NoC_space
                            height = self.ics if (yIdx + 1) % ycore == 0 else NoC_space
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics + self.core_w
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics + self.core_h
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
        outFile.close()
        output_file = os.path.join(self.path,'floorplan','interposer.flp')
        gen_cover_flp('interposer',sys_width, sys_height,eblk_w, eblk_h, output_file)
        output_file = os.path.join(self.path,'floorplan','dram.flp')
        gen_cover_flp('dram', sys_width, sys_height,eblk_w, eblk_h, output_file)
        self.sys_width = sys_width
        self.sys_height = sys_height
        self.eblk_w = eblk_w
        self.eblk_h = eblk_h
        return die_h, die_w, sys_height, sys_width
    
    def gen_sys_floorplan_nonunf(self, ics, xx, yy, xcut, ycut, NoC_space = 0.0002, sys_flp_name = 'output_3D.flp'):
        core_flp_path = os.path.join(self.path,'floorplan','single_core_3D.flp')
        sys_flp_path = os.path.join(self.path,'floorplan', sys_flp_name)
        with open(core_flp_path, 'r') as infile:
            lines = infile.readlines()
        blocks = []
        coreX = 0
        coreY = 0
        for line in lines:
            #   print(line)
            fields = line.strip().split('\t')
            if "#" not in line and len(fields) > 1:
                    name = fields[0]
                    # print(fields)
                    width = round(float(fields[1]),5) 
                    height = round(float(fields[2]),5)
                    leftX = round(float(fields[3]), 5)
                    bottomY = round(float(fields[4]), 5)
                    ## if the name is constructed with first '_'，the created dead block by hotfloorplan
                    if name[0] != '_':  
                        block_tmp = block_flp(name,width,height,leftX,bottomY)
                        blocks.append(block_tmp)
                    coreX = max(leftX + width,coreX)
                    coreY = max(bottomY + height,coreY)  
        # generate the entire floorplan with chiplet system
        outFile = open(sys_flp_path,'w')
        self.ics = ics
        xcore_list = [xx // xcut for _ in range(xcut)]
        ycore_list = [yy // ycut for _ in range(ycut)]
        die_h_list = []
        die_w_list = []
        for i in range(xx%xcut):
            xcore_list[i] +=1
        for i in range(yy%ycut):
            ycore_list[i] +=1
        for xcore in xcore_list:
            die_w = self.core_w * xcore + NoC_space * (xcore - 1)
            die_w_list.append(die_w)
        for ycore in ycore_list:
            die_h = self.core_h * ycore + NoC_space * (ycore - 1)
            die_h_list.append(die_h)
        sys_width  = sum(die_w_list) + self.ics * (xcut - 1)
        sys_height = sum(die_h_list )+ self.ics * (ycut - 1)
        # print(self.core_h, self.core_w , die_h, die_w, sys_height, sys_width)
        # print(die_h, die_w, sys_height, sys_width)
        # assert sys_height <= h_const, sys_width <= w_const
        # if sys_width < w_const or sys_height < h_const:
        #     eblk_w = (w_const - sys_width) / 2
        #     eblk_h = (h_const - sys_height) /2
        # else:
        #     eblk_w = 0.001
        #     eblk_h = 0.001
        eblk_w = 0.001
        eblk_h = 0.001
        str_  = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk0', eblk_w + sys_width, eblk_h , 0, 0)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk1', eblk_w, eblk_h + sys_height, 0, eblk_h)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk2', eblk_w + sys_width, eblk_h , eblk_w, eblk_h + sys_height)
        str_ += '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format('eblk3', eblk_w, eblk_h + sys_height, eblk_w + sys_width, 0)
        outFile.write(str_)
        for yi, ycore in enumerate(ycore_list):
            for xi, xcore in enumerate(xcore_list):
                xchipIdx, ychipIdx = xi, yi
                for cyi in range(ycore):
                    for cxi in range(xcore):
                        for n in range(len(blocks)):
                            block_tmp = blocks[n]
                            xchipIdx, ychipIdx = xi, yi
                            yIdx = sum(ycore_list[0: ychipIdx]) + cyi
                            xIdx = sum(xcore_list[0: xchipIdx]) + cxi
                            string_tmp = block_tmp.covert_chiplet(self.ics, NoC_space,xIdx, yIdx, self.core_w, self.core_h, xx, xchipIdx, ychipIdx, eblk_w, eblk_h)
                            outFile.write(string_tmp) 

                        idx = yIdx * xx + xIdx
                        if xIdx < xx -1:
                            name = 'blockX_{}'.format(idx)
                            width = self.ics if (xIdx + 1) == sum(xcore_list[0:xchipIdx+1]) else NoC_space
                            height = self.core_h
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics + self.core_w
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics
                            # str_ = name + '\t' + str(width) + '\t' + str(height) + '\t' + str(leftX) + '\t' + str(bottomY) + '\n'
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
                        if yIdx < yy - 1:
                            name = 'blockY_{}'.format(idx)
                            width = self.core_w
                            height = self.ics if (yIdx + 1) == sum(ycore_list[0:ychipIdx+1]) else NoC_space
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics      
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics + self.core_h
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
                        if xIdx < xx - 1 and yIdx < yy - 1:
                            name = 'blockXY_{}'.format(idx)
                            width = self.ics if (xIdx + 1) == sum(xcore_list[0:xchipIdx+1]) else NoC_space
                            height = self.ics if (yIdx + 1) == sum(ycore_list[0:ychipIdx+1]) else NoC_space
                            leftX = eblk_w + xIdx * self.core_w + NoC_space * (xIdx - xchipIdx) + xchipIdx * self.ics + self.core_w
                            bottomY = eblk_h + yIdx * self.core_h + NoC_space * (yIdx - ychipIdx) + ychipIdx * self.ics + self.core_h
                            str_ = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(name, width, height, leftX, bottomY)
                            outFile.write(str_)
        outFile.close()
        output_file = os.path.join(self.path,'floorplan','interposer.flp')
        gen_cover_flp('interposer',sys_width, sys_height,eblk_w, eblk_h, output_file)
        output_file = os.path.join(self.path,'floorplan','dram.flp')
        gen_cover_flp('dram', sys_width, sys_height,eblk_w, eblk_h, output_file)
        self.sys_width = sys_width
        self.sys_height = sys_height
        self.eblk_w = eblk_w
        self.eblk_h = eblk_h
        return die_h_list, die_w_list, sys_height, sys_width

    def run_hotspot(self, ptrace='cores_3D.ptrace', draw_fig=False):
        config_file = os.path.join(self.path,'example.config')
        flp_file = os.path.join(self.path,'floorplan','output_3D.flp')
        ptrace_file = os.path.join(self.path, 'ptrace', ptrace)
        shell = os.path.join(self.path,'run.sh')

        interposer_side = round(max(self.eblk_h * 2 + self.sys_height, self.eblk_w * 2 + self.sys_width),3) ## the aspect radio is 1, thus h == w
        # print(s_spreader, s_sink)
        command = [shell, config_file, flp_file, ptrace_file, str(interposer_side), self.path]
        run_command(command)
        if draw_fig:
            shell = os.path.join(self.path,'draw.sh')
            resolution = '128'
            run_command([shell, resolution, self.path + '/'+flp_file])
                          
                         
        



