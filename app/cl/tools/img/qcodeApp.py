import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image
from pyzbar.pyzbar import decode

def decode_qr(image_path):
    """识别图片中的二维码/条形码"""
    try:
        img = Image.open(image_path)
        decoded_objects = decode(img)
        if not decoded_objects:
            return None
        results = []
        for obj in decoded_objects:
            results.append({
                'type': obj.type,
                'data': obj.data.decode('utf-8')
            })
        return results
    except Exception as e:
        raise e

def select_image():
    """打开文件选择对话框并识别"""
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")]
    )
    if not file_path:
        return
    
    txt_result.delete(1.0, tk.END)
    txt_result.insert(tk.END, f"正在识别: {file_path}\n{'='*40}\n")
    
    try:
        results = decode_qr(file_path)
        if results is None:
            txt_result.insert(tk.END, "❌ 未检测到任何二维码或条形码。\n")
        else:
            for i, res in enumerate(results, 1):
                txt_result.insert(tk.END, f"✅ 结果 {i}:\n")
                txt_result.insert(tk.END, f"   类型: {res['type']}\n")
                txt_result.insert(tk.END, f"   内容: {res['data']}\n\n")
    except Exception as e:
        messagebox.showerror("错误", f"识别失败: {e}")

# 创建主窗口
app = tk.Tk()
app.title("二维码识别工具")
app.geometry("600x400")

frame = tk.Frame(app)
frame.pack(pady=10)

btn_select = tk.Button(frame, text="选择图片并识别", command=select_image, font=("Arial", 12))
btn_select.pack()

txt_result = scrolledtext.ScrolledText(app, wrap=tk.WORD, font=("Consolas", 10))
txt_result.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
txt_result.insert(tk.END, "点击上方按钮选择包含二维码的图片...\n")

app.mainloop()