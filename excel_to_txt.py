#-*- coding:UTF-8 -*-
import xlrd
def strs(row):
    """
    :返回一行数据
    """
    try:
        values = "";
        for i in range(len(row)):
            if i == len(row) - 1:
                values = values + str(row[i])
            else:
                #使用“，”逗号作为分隔符
                values = values + str(row[i]) + "," 
        return values
    except:
        raise
def xls_txt(xls_name,txt_name):
    """
    :excel文件转换为txt文件
    :param xls_name excel 文件名称
    :param txt_name txt   文件名称
    """
    try:
        data = xlrd.open_workbook(xls_name)
        sqlfile = open(txt_name, "a") 
        table = data.sheets()[0] # 表头
        nrows = table.nrows  # 行数
        #如果不需跳过表头，则将下一行中1改为0
        for ronum in range(1, nrows):
            row = table.row_values(ronum)
            values = strs(row) # 条用函数，将行数据拼接成字符串
            sqlfile.writelines(values) #将字符串写入新文件
        sqlfile.close() # 关闭写入的文件
    except:
        pass
if __name__ == '__main__':
    xls_name = 'G:/test.xls'
    txt_name = 'G:/test.txt'
    xls_txt(xls_name,txt_name)
