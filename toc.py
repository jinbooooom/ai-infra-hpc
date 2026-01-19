# -*- coding:utf-8 -*-

import sys
import os
from optparse import OptionParser

class fileConfigElement:
    def __init__(self, title="", file=""):
        self.title = title
        self.file = file

def githubUrl(url):
    # 将空格替换为 "-"
    url = url.replace(" ", "-")

    # 删除掉 specialChars 中的字符，不对 '-' 做任何操作
    englishSpecialChars = r'\/"<>@#$%^&*()+,.!:;`='
    chineseSpecialChars = r'，。、！（）？“”：；|'
    specialChars = englishSpecialChars + chineseSpecialChars
    for specialChar in specialChars:
        url = url.replace(specialChar, '')

    return url

if __name__ == "__main__":
    
    cudaConf = fileConfigElement("## CUDA 总结", "02hpc/05cuda/readme.md")
    topoConf = fileConfigElement("## 互联总结", "03link/03topo/readme.md")
    gpuDirectConf = fileConfigElement("## GPUDirect 总结", "03link/05gpuDirect/readme.md")
    rdmaConf = fileConfigElement("## RDMA 总结", "03link/08infiniband/rdma.md")
    ncclConf = fileConfigElement("## NCCL 总结", "05ccl/02nccl/readme.md")
    tarinAndInferConf = fileConfigElement("## 训练与推理总结", "06trainAndInfer/readme.md")

    rootReadmePath = "README.md"
    repoLink = "https://github.com/jinbooooom/ai-infra-hpc/blob/master/"
    confList = [cudaConf,
                topoConf,
                gpuDirectConf,
                rdmaConf,
                ncclConf,
                tarinAndInferConf
    ]

    saveTocPath = "jinbo_toc.md"
    with open(saveTocPath, 'w') as fw:
        for conf in confList:
            print(f"prepare process {conf.file}")
            fw.write(f"{conf.title}\n")
            filePath, fileName = os.path.split(conf.file)
            linkPrefix = repoLink + conf.file
            #print(linkPrefix)
            skip = 0
            with open(conf.file, 'r') as fr:
                for readLine in fr.readlines():
                    # 找到了代码块，就都跳过，不管以 # 开头的内容，因为这在代码区可能指的是注释而不是文本区的标题等级
                    if readLine.find("```") == 0:
                        skip = ~skip
                        continue
                
                    if (skip):
                        continue

                    if readLine[0] != "#":
                        continue

                    level = 0
                    while readLine[level] == "#":
                        level += 1

                    title_from = level
                    title_end = level
                    # 去掉末尾的换行和回车
                    while readLine[title_end] not in "\r\n":
                        title_end += 1
                    title = readLine[title_from : title_end]
                    # 去掉左右空格
                    title = title.strip()
                    # 处理含有超链接的标题 [title](link)
                    idx_start = title.find("[")
                    idx_end = title.find("](")
                    if idx_end != -1:
                        title = title[idx_start + 1 : idx_end]
                    # 将空格替换为%20 
                    #url = title.replace(' ' , '%20')
                    url = title
                    url = githubUrl(url)  

                    blank_str = ""
                    sign_str = "#"
                    for i in range(level - 1):
                        blank_str += "  "
                        sign_str += "#"
                    # 第一个无序列表不含有空格
                    blank_str = blank_str[2: ]
                    
                    title = f"[{title}]({linkPrefix}#{url})"
                    title = blank_str + " - " + title + '\n'
                    # print(title)
                    fw.write(title)

            print(f"process {conf.file} done!")

    bkRootReadmePath = "bk_" + rootReadmePath
    with open(bkRootReadmePath, 'w') as fw:
        print(f"extract the contents of the README.md")
        with open(rootReadmePath, 'r') as fr:
            #print(fr.readlines())
            for readLine in fr.readlines():
                fw.write(readLine)
                if readLine.find("---------------------") == 0:
                    print("find delimiter ---------------------")
                    break

        print(f"add contents of toc")
        with open(saveTocPath, 'r') as fr:
            for readLine in fr.readlines():
                fw.write(readLine)

    print("remove and rename file")
    if os.path.exists(saveTocPath): 
        os.remove(saveTocPath)
    if os.path.exists(rootReadmePath): 
        os.remove(rootReadmePath)
    if os.path.exists(bkRootReadmePath):
        os.rename(bkRootReadmePath, rootReadmePath)
    print("new README.md was generated, everything is done!")
    

    



        