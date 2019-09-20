import sys


ifile = sys.argv[1]
#print("convert:",ifile)

with open(ifile,"r") as f:
    f_line = f.readlines()
    i = 0
    while i < len(f_line): 
      each = f_line[i].rstrip()
      if each[-3:] == "jpg":
         f_name = each[:-4]
         i += 1
         num = int(f_line[i].rstrip())
         i += 1
         j = 0
         ret = []
         while j < num:
            buf_list = f_line[i + j].split()
            buf_list = [float(x) for x in buf_list]
            ret += [buf_list[0],buf_list[1],buf_list[2]+buf_list[0],buf_list[3]+buf_list[1]]
            j += 1
         i += num
         ret = [str(x) for x in ret]
         print("%s %s" % (f_name," ".join(ret)))
