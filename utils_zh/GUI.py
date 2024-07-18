import tkinter as tk
from tkinter import messagebox, filedialog
from utils_zh.predict import load_image
from PIL import Image, ImageTk
from googletrans import Translator


def GUI():

    def click_submit():
        image_path = click_submit.image_path
        # 将图像的路径加载到image_path中

        result = load_image(image_path)
        # 把路径放到函数中进行判断

        translator = Translator()
        result = translator.translate(result, src='en', dest='zh-cn').text
        # 将判断后的结果用Google的API进行转译

        messagebox.showinfo("该项目为:", result)
        # 将结果显示在弹窗中

    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[('Image Files',"*.jpg;*.png")])
        # 打开文件选择对话框 并让用户选择文件 且只显示.jpg .png文件

        if file_path:
            click_submit.image_path = file_path
            # 如果选择了文件 则将该路径保存 方便click_submit函数使用

            img = Image.open(file_path)
            # 将选择的文件路径上的图片加载到img中

            img = img.resize((227,227))
            # 设定显示的大小

            img = ImageTk.PhotoImage(img)
            # 将PIL图像转换为tkinter可用的PhotoImage对象

            image_label.config(image=img)
            # 将image_label的标签图像更新为img

            image_label.image = img
            # 保持对该图像的引用 直到下一次被替换

    root = tk.Tk()
    root.title('体育项目识别')
    root.geometry('400x380+500+250')
    # 定义主窗口 title名称 大小 距离右上角的距离

    upload_button = tk.Button(root,text='点击上传图片',width=10,height=3,command=upload_image)
    upload_button.pack(pady=10)
    # 创建一个按钮 当点击按钮时调用upload_image函数
    # 并设置这个按钮的显示文字 大小 与垂直边距

    image_label = tk.Label(root)
    image_label.pack()
    # 定义一个标签 用于显示图片

    click_submit = tk.Button(root,text='点击识别',width=7,height=2,command=click_submit)
    click_submit.pack(pady=10)
    # 创建一个按钮 当点击按钮时调用click_submit函数

    root.mainloop()
    # 启动Tkinter主事件循环 这个窗口开始运行并等待交互
