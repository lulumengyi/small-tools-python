from PIL import Image
def Images_Pdf(filename, output):
    images = []
for file in filename:
        im = Image.open(file)
        im = im.convert('RGB')
        images.append(im)
    
    images[0].save(output, save_all=True, append_images=images[1:])
Images_Pdf(["test1.jpg", "test2.jpg", "test3.jpg"], "output.pdf")

作者：Jackpop
链接：https://www.zhihu.com/question/23188097/answer/2517236366
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
