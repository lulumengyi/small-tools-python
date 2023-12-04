
目录
收起
Outline
一、组织文件常用模块
二、常见文件操作
1.获取目录列表
2 获取文件属性
3 创建目录
4 文件名匹配
5 创建临时文件和目录
6 删除文件和目录
7 复制、移动和重命名文件和目录
8 压缩与解压缩
三、应用实例
1 根据文件名（关键字匹配）分类
2 根据文件名中关键字改名字
3 压缩文件
4 对文件夹下的文件按文件名中的数字排序
5 显示某文件夹下子文件夹的大小
6 m3u8文件转MP4
7 logging日志捕获代码异常（traceback）
8 进度条
一、组织文件常用模块
二、常见文件操作
1.获取目录列表
1.1 使用os.listdir()：
1.2 使用os.scandir()：
1.3 使用pathlib.Path()：
2 获取文件属性
2.1 使用os.stat()：
2.2 使用os.scandir()：
2.3 使用pathlib.Path()：
3 创建目录
3.1 创建单个目录
3.1.1 使用os.mkdir()：
3.1.2 使用pathlib的mkdir函数：
3.2 创建多个目录（一创一路）
3.2.1 使用os.mkdir()：
3.2.2 使用pathlib的mkdir函数：
4 文件名匹配
4.1 endswith() 和 startswith() 字符串方法
4.2 使用fnmatch模块
4.3 使用glob模块
4.4 pathlib.Path.glob()
4.5 正则
4 遍历目录
5 创建临时文件和目录
5.1 使用TemporaryFile创建临时文件
5.2 使用TemporaryDirectory创建临时文件夹
6 删除文件和目录
6.1 删除单个文件
6.2 删除目录
7 复制、移动和重命名文件和目录
7.1 复制、移动文件
7.2 复制、移动目录
7.3 重命名文件和目录
8 压缩与解压缩
三、应用实例
1 根据文件名（关键字匹配）分类
2 根据文件名中关键字改名字
3 压缩文件
4 对文件夹下的文件按文件名中的数字排序
5 显示某文件夹下子文件夹的大小
6 m3u8文件转MP4
7 logging日志捕获代码异常（traceback）
8 进度条
Reference
Outline
一、组织文件常用模块
二、常见文件操作
1.获取目录列表
2 获取文件属性
3 创建目录
4 文件名匹配
5 创建临时文件和目录
6 删除文件和目录
7 复制、移动和重命名文件和目录
8 压缩与解压缩
三、应用实例
1 根据文件名（关键字匹配）分类
2 根据文件名中关键字改名字
3 压缩文件
4 对文件夹下的文件按文件名中的数字排序
5 显示某文件夹下子文件夹的大小
6 m3u8文件转MP4
7 logging日志捕获代码异常（traceback）
8 进度条


主要是文件夹及其内容管理操作，即只组织文件，并不对文件内容进行读写操作。

一、组织文件常用模块
os(Operating System):提供了一种方便地使用操作系统函数的方法，使代码通用于各个操作系统。
sys(System):可访问由解释器使用或维护的变量和与解释器交互的函数，用于操控Python的运行环境。
shutil(sh+util即shell+utility):是对OS中文件操作的补充,是高级的文件、文件夹以及压缩包处理模块，可实现复制、移动、改名、删除、打包、压缩、解压等功能。
pathlib：Python3.4新增标准库，使用面向对象的编程方式来表示文件系统路径，相比os.path，pathlib模块的 Path 对路径的操作更简单。
zipfile：可创建、打开以及提取ZIP文件。
tarfile：可创建、打开以及提取TAR文件。
py7zr：可创建、打开以及提取7Z文件，与zipfile、tarfile不同，这是第三方库，需要先安装才能使用。
他们的常见应用见下面二。

二、常见文件操作
本来想自己总结的，但是一搜发现有一篇总结的很棒的文章（下面参考链接1），所以这里就简单摘抄一下，然后补充一下。

1.获取目录列表
列出目录内容和过滤结果,常见有3中方法实现：
os.listdir() 返回一个Python列表，其中包含path参数所指目录的文件和子目录的名称。
os.scandir() （需Python 3.5+） 调用时返回一个迭代器而不是一个列表，并且os.scandir() 可以和with语句一起使用，因为它支持上下文管理协议，使用上下文管理器关闭迭代器并在迭代器耗尽后自动释放获取的资源。
pathlib.Path() （需Python 3.4+） 返回的是 PosixPath 或 WindowsPath 对象，这取决于操作系统。然后pathlib.Path() 对象有一个 .iterdir() 的方法用于创建一个迭代器包含该目录下所有文件和目录。由 .iterdir() 生成的每个条目都包含文件或目录的信息。

1.1 使用os.listdir()：
import os    
basepath = 'my_directory'    
for entry in os.listdir(basepath):    
    if os.path.isfile(os.path.join(base_path, entry)):    
        print(entry)
需多次调用 os.path,join()。

1.2 使用os.scandir()：
import os    

basepath = 'my_directory'    
with os.scandir(basepath) as entries:    
    for entry in entries:    
        if entry.is_file():    
            print(entry.name)
1.3 使用pathlib.Path()：
from pathlib import Path    

basepath = Path('my_directory')    
for entry in basepath.iterdir():    
    if entry.is_file():    
        print(entry.name)
如果将for循环和if语句组合成单个生成器表达式，则上述的代码可以更加简洁：

from pathlib import Path    

basepath = Path('my_directory')    
files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())    
for item in files_in_basepath:    
    print(item.name)
2 获取文件属性
Python可以很轻松的获取文件大小和修改时间等文件属性。可以通过使用 os.stat() ， os.scandir() 或 pathlib.Path 来获取。
os.scandir() 和 pathlib.Path() 能直接获取到包含文件属性的目录列表。这可能比使用 os.listdir() 列出文件然后获取每个文件的文件属性信息更加有效。
stat 结构:
st_mode : inode 保护模式
st_ino : inode 节点号。
st_dev : inode 驻留的设备。
st_nlink : inode 的链接数。
st_uid : 所有者的用户ID。
st_gid : 所有者的组ID。
st_size : 普通文件以字节为单位的大小；包含等待某些特殊文件的数据。
st_atime : 上次访问的时间。
st_mtime : 最后一次修改的时间。
st_ctime : 由操作系统报告的"ctime"。在某些系统上（如Unix）是最新的元数据更改的时间，在其它系统上（如Windows）是创建时间（详细信息参见平台的文档）。

2.1 使用os.stat()：
import os    

for entry in os.listdir('my_directory'):    
    info = os.stat(os.path.join(base_path, entry))    
    print(info.st_mtime)
2.2 使用os.scandir()：
import os    

with os.scandir('my_directory') as entries:    
    for entry in entries:    
        info = entry.stat()    
        print(info.st_mtime)
2.3 使用pathlib.Path()：
from pathlib import Path    

basepath = Path('my_directory')    
for entry in basepath.iterdir():    
    info = entry.stat()    
    print(info.st_mtime)
3 创建目录
3.1 创建单个目录
可使用os.mkdir()和pathlib的mkdir函数。
如果该目录已经存在，则都会抛出 FileExistsError 异常，可以使用异常处理，且pathlib的mkdir函数还可以传入 exist_ok=True 参数来忽略 FileExistsError 异常。

3.1.1 使用os.mkdir()：
import os    

os.mkdir('example_directory')
3.1.2 使用pathlib的mkdir函数：
from pathlib import Path    

p = Path('example_directory')    
try:    
    p.mkdir()    
except FileExistsError as e:    
    print(e)
或者忽略 FileExistsError 异常：

from pathlib import Path    

p = Path('example_directory')    
p.mkdir(exist_ok=True)
3.2 创建多个目录（一创一路）
可使用os.makedirs()和pathlib的mkdir函数（parents=True即可）。

3.2.1 使用os.mkdir()：
import os    

os.makedirs('2018/10/05', mode=0o770)
后面mode是指定该文件的权限。

3.2.2 使用pathlib的mkdir函数：
from pathlib import Path    

p = Path('2018/10/05')    
p.mkdir(parents=True, exist_ok=True)
指定parents=True即可，同时指定exist_ok=True当文件夹已存在时进行覆盖。

4 文件名匹配
获取目录中的文件列表后，可以通过文件名来搜索和查找文件，有以下几种方法：
endswith() 和 startswith() 字符串方法
fnmatch.fnmatch()
glob.glob()
pathlib.Path.glob()
* 正则

4.1 endswith() 和 startswith() 字符串方法
import os    

for f_name in os.listdir('some_directory'):    
    if f_name.endswith('.txt'):    
        print(f_name)
还可以通过列表解析是代码更简洁：

import os    

end_txt = [txt for txt in os.listdir('some_directory') if txt.endwith('.txt')]    
for end_txt_file in end_txt:    
        print(end_txt_file)
4.2 使用fnmatch模块
比字符串匹配强大，但比正则要弱，因为它只使用通配符。适用于简单场景。
通配符如下：
*：0或多个任意字符
？：1个任意字符
[]/[!]/[-]:同正则
fmatch模块常用方法如下：
filter(names,pattern):dui names列表进行过滤，匹配到的结果以list形式返回
fnmatch(filename,pattern):判断filename是否和pattern匹配
fnmatchcase(filename,pattern):功能同fnmatch,区别是它区分大小写
translate(pattern):将通配符pattern转化为正则表达式
使用示例：

import os    

end_txt = [txt for txt in os.listdir('some_directory') if fnmatch.fnmatch(txt, 'data_*_backup.txt')]    
for end_txt_file in end_txt:    
        print(end_txt_file)
4.3 使用glob模块
使用方法为：glob.glob(pathname)，其中pathname是用于指定文件路径，且其中可以含通配符。结果以list形式返回。
在文件路径中使用通配符在shell中很常见，通过glob，windows下也可以这样筛选文件了。

import glob    

for name in glob.glob('*[0-9]*.txt'):    
    print(name)
另一个常使用的例子是将其用于根据文件属性排序,即先通过glob筛选出一部分，然后通过sort排序：

import glob    

path = glob.glob('C:/users/dir/*.txt')    
sorted(path)                         #按名称排序    
sorted(path,key=os.path.getmtime)    #按修改时间排序    
sorted(path,key=os.path.getsize)     #按大小排序
还可通过glob.iglob()方便地实现在指定目录中递归搜索：

import glob    

for name in glob.iglob('**/*.py', recursive=True):    
    print(name)
4.4 pathlib.Path.glob()
pathlib.Path基本类似glob模块中的glob，事实上，pathlib 混合了许多 os ， os.path 和 glob 模块的最佳特性到一个模块中，这使得使用起来很方便。

from pathlib import Path    

p = Path('.')    
for name in p.glob('*.p*'):    
    print(name)    
for name in p.rglob('*.p*'):#.rglob实现递归    
    print(name)
4.5 正则
正则是最强的搜索手段，当上述方法无法使用时，就可考虑使用正则。
比较常用的是re.match(pattern,string,flags)和re.search(pattern,string,flags),区别是match是从头开始匹配pattern,若不匹配则直接失败，而search则是搜索整个字符串，一旦匹配到一个就成功，并返回结果。
所以match中写正则要从头写到尾，即全匹配；而search的正则可以只写关心的部分，即关键字匹配。示例如下：
使用match：

import re    
import glob    

file_list  = [f for f in glob.glob('*.v')+glob.glob('*.sv') if not re.match(r'DUT.sv', f)]
使用seatch：

import os    
import re    
import glob    

for file in os.listdir('.'):    
    if re.search('苹果|香蕉',file):    
        category = "水果"
4 遍历目录
主要是应用os.walk()函数。

os.walk(top,topdown=True,onerror=None,fllowlinks=False)
top:顶层目录
topdown:自顶而下遍历，即先遍历根目录再遍历子目录，默认开。若改为False则先遍历子目录。在递归删除文件夹或文件时需要改为自下而上，即先删除子目录。
fllowlinks：当存在软连接目录时，默认是不访问。想访问改为True即可。
onerror：当walk()执行出现异常时调用它，我们的段位目前基本用不到。
它返回的是生成器，也即需要不断遍历它（for循环即可）才能得到所有内容，且它每次返回的对象都是一个三元组(root,dirs,files)：
root:当前正在遍历的文件夹的地址，常用于搭配dirs和files使用，指出其中的文件夹/文件的父路径；单独使用时是指包含top文件夹及其所有子文件夹在内的一个列表。
dirs：当前正在遍历的文件夹下所有的子文件夹，列表形式。
files：当前正在遍历的文件夹下所有的文件，列表形式。
下面通过两个例子来说明这些参数如何应用：
eg1:保持目录 a 的目录结构，在 b 中创建对应的文件夹,并把a中所有的文件加上后缀 _bak（源于下面参考链接4）

import os    

Root = 'a'    
Dest = 'b'    

for (root, dirs, files) in os.walk(Root):    
    new_root = root.replace(Root, Dest, 1)    
    if not os.path.exists(new_root):    
        os.mkdir(new_root)    

    for d in dirs:    
        d = os.path.join(new_root, d)    
        if not os.path.exists(d):    
            os.mkdir(d)    

    for f in files:    
        # 把文件名分解为 文件名.扩展名    
        # 在这里可以添加一个 filter，过滤掉不想复制的文件类型，或者文件名    
        (shotname, extension) = os.path.splitext(f)    

        # 原文件的路径    
        old_path = os.path.join(root, f)    
        new_name = shotname + '_bak' + extension    

        # 新文件的路径    
        new_path = os.path.join(new_root, new_name)    
        try:    
            # 复制文件    
            open(new_path, 'wb').write(open(old_path, 'rb').read())    
        except IOError as e:
eg2:查找文件，并打印结果全路径（也是抄的，但找不到在哪抄的了...）

import os    

for relpath,dirs,files in os.walk(top_path):    
    if name in file:    
        full_path = os.path.join(top_path,relpath,name)    
        full_path = os.path.normpath(os.path,abspath(full_path))    
        print(full_path)
5 创建临时文件和目录
Python提供了 tempfile 模块来便捷的创建临时文件和目录。tempfile 可以在你程序运行时打开并存储临时的数据在文件或目录中。 tempfile 会在程序停止运行后删除这些临时文件。
原本以为这在需要产生中间文件的脚本中非常有用，但真正使用起来还是创一个临时文件最后再删掉来得简单（见后面那个MarkDown规范脚本）。

5.1 使用TemporaryFile创建临时文件
from tempfile import TemporaryFile    

# 创建一个临时文件并为其写入一些数据    
fp = TemporaryFile('w+t')    
fp.write('Hello World!')    

# 回到开始，从文件中读取数据    
fp.seek(0)    
data = fp.read()    
print(data)    

# 关闭文件，之后他将会被删除    
fp.close()
值得注意的是，临时文件是不能命名的，若需要命名使用 tempfile 生成的临时文件，则需使用 tempfile.NamedTemporaryFile()。
此外，.TemporaryFile() 也是一个上下文管理器，因此它可以与with语句一起使用。 使用上下文管理器会在读取文件后自动关闭和删除文件：

from tempfile import  TemporaryFile    

with TemporaryFile('w+t') as fp:    
    fp.write('Hello universe!')    
    fp.seek(0)    
    fp.read()    
# 临时文件现在已经被关闭和删除
5.2 使用TemporaryDirectory创建临时文件夹
import tempfile    
import os    

tmp = ''    

with tempfile.TemporaryDirectory() as tmpdir:    
    print('Created temporary directory ', tmpdir)    
    tmp = tmpdir    
    print(os.path.exists(tmpdir))    

print(tmp)    
print(os.path.exists(tmp))
6 删除文件和目录
可以使用 os，shutil 和 pathlib 模块中的方法删除单个文件，目录和整个目录树。

6.1 删除单个文件
要删除单个文件，可使用 pathlib.Path.unlink()，os.remove() 或 os.unlink()。
os.remove() 和 os.unlink() 在语义上是相同的。

import os    

data_file = 'C:\\Users\\vuyisile\\Desktop\\Test\\data.txt'    
os.remove(data_file)
import os    

data_file = 'C:\\Users\\vuyisile\\Desktop\\Test\\data.txt'    
os.unlink(data_file)
在文件上调用 .unlink() 或 .remove() 会从文件系统中删除该文件。 如果传递给它们的路径指向目录而不是文件，这两个函数将抛出 OSError 。 为避免这种情况，可以检查你要删除的内容是否是文件，并在确认是文件时执行删除操作，或者可以使用异常处理来处理 OSError ：

import os    

data_file = 'home/data.txt'    
# 使用异常处理    
try:    
    os.remove(data_file)    
except OSError as e:    
    print(f'Error: {data_file} : {e.strerror}')#打印f-string格式化
还可以使用 pathlib.Path.unlink() 删除文件：

from pathlib import Path    

data_file = Path('home/data.txt')    
try:    
    data_file.unlink()    
except IsADirectoryError as e:    
    print(f'Error: {data_file} : {e.strerror}')
6.2 删除目录
可以使用os.rmdir()、pathlib.Path.rmdir()以及shutil.rmtree()，前两个只在删除空目录的时候有效，如果目录不为空，则会抛出 OSError。

import os    

trash_dir = 'my_documents/bad_dir'    
try:    
    os.rmdir(trash_dir)    
except OSError as e:    
    print(f'Error: {trash_dir} : {e.strerror}')
from pathlib import Path    

trash_dir = Path('my_documents/bad_dir')    
try:    
    trash_dir.rmdir()    
except OSError as e:    
    print(f'Error: {trash_dir} : {e.strerror}')
以及递归删除所有空文件夹：

import os    

for dirpath, dirnames, files in os.walk('.', topdown=False):#topdown=False    
    try:    
        os.rmdir(dirpath)    
    except OSError as ex:#如果目录不为空，则引发OSError并跳过该目录    
        pass
要删除非空目录和完整的目录树，Python提供了 shutil.rmtree() :

import shutil    

trash_dir = 'my_documents/bad_dir'    
try:    
    shutil.rmtree(trash_dir)    
except OSError as e:    
    print(f'Error: {trash_dir} : {e.strerror}')
7 复制、移动和重命名文件和目录
复制和移动时，对于目标文件已存在的处理不同（赋值和移动不同，文件和文件夹的处理也不一样），有需要最好先判断目标文件是否存在，分情况做处理。

7.1 复制、移动文件
复制文件可以用shutil.copy() 和 shutil.copy2()，二者区别在于shutil.copy2()在复制时保留所有文件元数据（如文件的创建和修改时间等）。
shutil.copy('文件名',‘目的文件夹) 复制到目的文件夹下面
shutil.copy('文件名',‘目的文件夹/新文件名’) 复制到目的文件夹下面，之后重命名。若新文件名已存在，则该文件的内容将被覆盖。

import shutil    

src = 'path/to/file.txt'    
dst = 'path/to/dest_dir'    
shutil.copy(src, dst)
移动文件 shutil.move()，若目标文件已存在，则会报错。
shutil.move('文件名',‘目的文件夹’) 移动到目的文件夹下面
shutil.move('文件名',‘目的文件夹/新文件名’) 移动到目的文件夹下面，之后重命名。

7.2 复制、移动目录
使用shutil.copytree() 复制整个目录及其中包含的所有内容。 shutil.copytree(src，dest) 接收两个参数：源目录和将文件和文件夹复制到的目标目录。
若目标目录下有同名文件则会报错。

import shutil    

shutil.copytree('要复制的文件夹', '目的文件夹/要复制的文件夹') #复制到目的文件夹下面    
shutil.copytree('要复制的文件夹', '目的文件夹/新文件夹') #复制过去并重命名为 新文件夹
移动文件夹和移动文件一样，也使用shutil.move('要复制的文件夹', '目的文件夹')。移动到目标文件夹下，结构变为目的文件夹/要复制的文件夹，若目的文件夹下有同名文件则报错；若目的文件夹不存在，则直接将要赋值的文件夹改名为目的文件夹。

7.3 重命名文件和目录
上面在移动和复制时也可以实现重命名，不过Python有专门的函数os.rename(src，dst)用于重命名文件和目录。也可以用pathlib 模块中的 rename（），与os.rename类似。

import os    

os.rename('first.zip', 'first_01.zip')
from pathlib import Path    

data_file = Path('data_01.txt')    
data_file.rename('data.txt')
8 压缩与解压缩
python自带的库可以操作rar与zip格式，安装py7zr库可以操作7z格式。它们使用起来都很类似，都先创建一个对象（创建时传很多参数以指定压缩/解压、密码、模式等等），并且这个对象可以像大多数文件的对象一样打开，然后调用各自的方法即可。
好像BandZip有提供命令行访问，可以用windows的命令行，也可以用python，但是我试了半天没调成功，最后用的py7zr。
（PS：其实我觉得好压挺好用的，就是老弹广告，卸载时还解释广告弹窗不是它的锅，然而卸了之后再也没弹过...)

三、应用实例
1 根据文件名（关键字匹配）分类
from pathlib import Path    
import os    
import re    
import shutil    

def sort(top_path,out_top_path):    
    category_list = ["01.苹果","02.桃","03.梨","04.其他"]    
    for category in category_list:    
        path = Path( out_top_path ) / category    
        if not path.exists():    
            path.mkdir(parents=True)    

    for (root,dirs,files) in os.walk(top_path):    
        if not re.search('统计',root):    
            for file in files:    
                if re.match('.*\.txt',file):    
                    if re.search('苹果|秦冠|金元帅',file):    
                        category = "01.苹果"    
                    elif re.search('黄桃|血桃',file):    
                        category = "02.桃"    
                    elif re.search('砀山|梨',file):    
                        category = "03.梨"    
                    else:    
                        category = "04.其他"    

                    if not os.path.exists(os.path.join(out_top_path,category,file)):    
                        shutil.move(os.path.join(root,file),os.path.join(out_top_path,category))
对top_path下所有层次的文件进行分类，并将分类结果移动到out_top_path下。

2 根据文件名中关键字改名字
import os    
import re    

def rename(top_path):    
    for (root,dirs,files) in os.walk(top_path):    
        for file in files:    
            if re.match('.*\.txt',file):    
                old_name = os.path.join(root,file)    
                (new_name,extension) = os.path.splitext(file)    
                new_name = re.sub("pingguo|apple", "苹果", new_name)    
                new_name = re.sub("peach", "桃", new_name)    
                new_name = re.sub("pear", "桃", new_name)    

                new_name = os.path.join(root,new_name+extension)    
                try:    
                    os.rename(old_name,new_name)    
                except:    
                    continue
对top_path下所有文件应用改名规则，若目标已存在则利用错误处理机制跳过。

3 压缩文件
from pathlib import Path    
import os    
import re    
import py7zr    

def compress(source_file,destination_file):    
    password = "123456"    

    destination_dir = Path(destination_file).resolve()/"压缩包"    
    if not destination_dir.exists():    
            destination_dir.mkdir(parents=True)    

    for (root,dirs,files) in os.walk(source_file):    
        for file in files:    
            if re.match('.*\.txt',file):    
                file_path = Path(root).resolve()/file    
                with py7zr.SevenZipFile(destination_dir/(file_path.stem+".7z"), mode='w',password=password) as archive:    
                    archive.write(Path(root).resolve()/file,file)    
                print("{} \thave been compressed successfully!".format(Path(root).resolve()/file))
对source_file下所有文件进行压缩，带密码，且一个文件一个同名压缩包，并且压缩包存入destination_file/压缩包中。

4 对文件夹下的文件按文件名中的数字排序
使用shell时曾纠结过这个问题，后来用shell的sort解决了：
alias lln 'll | sort -k9,9 -V | cat -n'
alias lsn 'ls | sort -V | cat -n'
现在用python也是有办法实现的，也是sort函数 ：)

import os    
file_path = os.path.abspath('.')    
boxer        = os.listdir(file_path)    
boxer.sort(key=lambda fname:int(fname.split('.')[0][4:]))
适用于文件名有规律且数字的位置已知，若有规律但数字的位置不已知，则可以通过程序解析数字的位置，然后再排序，见下面m3u8文件转mp4的脚本。

5 显示某文件夹下子文件夹的大小
这个在shell中也很好实现alias duh 'du --max-depth=1 -h',python的话得写个函数。

from pathlib import Path    
import os    

def show_dir(top_path):    
    path = Path(top_path)    
    dir_size = {}    

    # calc dir size    
    for iterdir in path.iterdir():    
        if iterdir.is_dir:    
            size = 0    
            for (root,dirs,files) in os.walk(iterdir):    
                size += sum([os.path.getsize(os.path.join(root, file)) for file in files])    
            dir_size[str(iterdir)] = size/1024/1024    

    # sort    
    dir_size = sorted(dir_size.items(),key=lambda dir_size:dir_size[1],reverse=True)    

    # print    
    for dir,size in dir_size:    
        print('{:<15}: {:>7.2f} Mb'.format(dir,size))
6 m3u8文件转MP4
m3u8转mp4的原理就是将m3u8按顺序拼接起来即可，这个顺序就是m3u8文件名中数字的顺序。
对于一个特定的文件m3u8文件夹，其中的文件名数字位置是已知的，但想要脚本更通用，就得让脚本自动解析文件名中数字的位置。实现如下：

import sys    
import os    
import re    
import glob    
import shutil    
import numpy as np    
import logging    
import time    

def merge_ts(file_path,o_file_path):    
    boxer        = os.listdir(file_path)    

    # suffix    
    boxer_suffix = [os.path.splitext(file)[1] for file in boxer]    
    file_suffix  = max(boxer_suffix,key=boxer_suffix.count)    
    boxer  = [file for file in boxer if(os.path.splitext(file)[1] == file_suffix)]    

    # to find mean of boxer    
    file_num_bits = len(str(len(boxer)))    
    avg   = int(np.ceil(np.mean([len(str(file)) for file in boxer]) if len(boxer) > 0 else 0))    
    boxer = list(set(boxer).difference(set([file for file in boxer if( (abs(len(str(file)) - avg ) > file_num_bits) or (file.startswith('.')))])))    
    avg     = int(np.ceil(np.mean([len(str(file.split('.')[0])) for file in boxer]) if len(boxer) > 0 else 0))    
    boxer   = list(set(boxer).difference(set([file for file in boxer if abs(len(str(file.split('.')[0])) - avg ) > (file_num_bits - 1)])))    

    # sort    
    file_num_bits = len(str(len(boxer)-1))    
    max_len = max([len(str(file.split('.')[0])) for file in boxer]) if len(boxer) > 0 else 0    
    boxer   = [file for file in boxer if re.match('^[0-9]',file.split('.')[0][max_len-file_num_bits:])]    
    file_num_bits = len(str(len(boxer)-1))    
    max_len = max([len(str(file.split('.')[0])) for file in boxer]) if len(boxer) > 0 else 0    
    boxer.sort(key=lambda fname:int(fname.split('.')[0][max_len-file_num_bits:]))    

    # prefix    
    prefix = boxer[0].split('.')[0][:max_len-file_num_bits]    
    if( prefix != '' and (len(boxer) > 500) ):    
        for file in boxer:    
            os.rename(file,file.split('.')[0][max_len-file_num_bits:])    
        boxer = [file.split('.')[0][max_len-file_num_bits:] for file in boxer]    

    # cmd    
    cmd_str    = '+'.join(boxer)    
    exec_str   = "copy /b " + cmd_str + ' ' + o_file_path + '>nul'    
    print("copy /b " + cmd_str + ' ' + o_file_path)    
    os.system(exec_str)    
# --Main-----------------------------------------------------------------------------------------------------------    
if __name__=='__main__':    
    top_path = os.path.abspath('.')    

    for (root,dirs,files) in os.walk(top_path):    
        # 进度条    
        barLen   = 20  # 进度条的长度    
        task_num = len(dirs)    
        done_num = 0    
        t0 = time.time()    

        for dir in dirs:    
            if not dir.startswith('.'):    
                os.rename(dir,dir.replace(' ',''))    
                file_path   = os.path.join(root,dir)    
                o_file_path = os.path.join(top_path,dir.split('.')[0]+'.mp4')    
                os.chdir(file_path)    
                try:    
                    boxer = merge_ts(file_path,o_file_path)    
                except Exception as e:    
                    logging.exception(e)    
                    continue    
                os.chdir(top_path)    
                shutil.move(os.path.join(root,dir),os.path.join('..','M3U8-Done'))    

                # 进度条    
                done_num += 1    
                perFin = done_num/task_num    
                numFin = round(barLen*perFin)    
                numNon = barLen-numFin    
                runTime = time.time() - t0    
                print(    
                    f"{done_num:0>{len(str(task_num))}}/{task_num}",    
                    f"|{'█'*numFin}{' '*numNon}|",    
                    f"任务进度: {perFin*100:.2f}%",    
                    f"已用时间: {runTime:.2f}S",    
                    end='\r')
因为M3U8文件夹下有时还会有一些干扰项，所以通过文件名平均长度将其排除，并解析前缀和后缀，并据此再排除一批干扰项，最后只剩下含数字序号的有规则的m3u8文件，合并即可。
当m3u8文件数量过多时就要考虑将文件名简化一下，否则传给CMD的命令行太长它会拒绝执行。
其中还有错误处理机制和进度条，这两个见下面。

7 logging日志捕获代码异常（traceback）
当批处理一堆文件时，可能我们挂上脚本就跑了，所以当某一个文件处理出错时我们会比较希望记录下出错的文件和出错的信息，但不希望脚本停止运行。这点靠错误处理机制就能实现，简单版如下：

try:    
    pass              #实际代码部分    
except Exception as e:#Except是通用异常类型    
    print(e)
这样可以实现上述功能（不加continue也可以的），但print(e)打印出的信息有限，我们希望打印得和直接停止运行时打印的信息一样。
实际上，程序在执行过程中发生异常就会中断程序，调用python默认的异常处理器，并在终端输出异常信息，这个异常信息时由traceback模块跟踪异常所返回的信息。
现在我们通过try....except....来处理异常（即自定义的异常处理器），也是可以记录traceback信息的，这要用到logging模块。

try:    
    pass                    #实际代码部分    
except Exception as e:      #Except是通用异常类型    
    logging.exception(e)    #等价于logging.error(e,exc_info=True)
这里主要涉及python错误处理机制和logging模块的使用，参见下面相关参考链接。

8 进度条
好像有专门的进度条库，非官方的，需要安装，所以这里就选择了自己写代码的简单版本。
参考大佬的代码改的，原理也很简单，上来先统计总共多少个文件夹（待处理对象）即总任务数，然后每处理完一个就计算一下进度，并打印一次进度条。

top_path = os.path.abspath('.')    
for (root,dirs,files) in os.walk(top_path):    
    # 进度条    
    barLen   = 20  # 进度条的长度    
    task_num = len(dirs)    
    done_num = 0    
    t0 = time.time()    

    for dir in dirs:    
        pass#实际代码    

        # 进度条    
        done_num += 1    
        perFin = done_num/task_num    
        numFin = round(barLen*perFin)    
        numNon = barLen-numFin    
        runTime = time.time() - t0    
        print(    
            f"{done_num:0>{len(str(task_num))}}/{task_num}",    
            f"|{'█'*numFin}{' '*numNon}|",    
            f"任务进度: {perFin*100:.2f}%",    
            f"已用时间: {runTime:.2f}S",    
            end='\r')


