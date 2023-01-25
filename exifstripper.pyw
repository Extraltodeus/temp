from tkinter import *
from PIL import Image, ImageTk
import os
import sys
import threading
from time import sleep
import subprocess
import pyperclip
from pathlib import Path

join = os.path.join

def daemonizer(fName, *args):
    try:
        daemon = threading.Thread(target=fName, args=args)
        daemon.daemon = True
        daemon.start()
        return daemon
    except Exception as e:
        print(e)

# sys.stdin.reconfigure(encoding='utf-8')
# sys.stdout.reconfigure(encoding='utf-8')

with open("./models.txt") as file:
    models = [line.rstrip() for line in file]

img_formats = [".png",".jpg",".jpeg"]
max_size = 512
current_dir = "./"
selected_folder = "."
current_subdirs = [ name for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name)) ]
# current_dir = os.listdir("./")
# print(current_subdirs)
#
# from pathlib import Path
# sorted(Path('.').iterdir(), key=lambda f: f.stat().st_mtime)

def get_infos(path):
    image = Image.open(path)
    i = image.info
    if "parameters" in i: return i["parameters"]
    else: return "No description."

def refresh_stringvar(files_path_name):
    global file_names
    file_names.set([i for i in files_path_name])
    listbox.config(listvariable=file_names)

def refresh_files(current_subdirs):
    global files_path_name
    files_path_name = {}
    for dir in current_subdirs:
        if dir == selected_folder:
            # files = sorted(Path(dir).iterdir(), key=os.path.getmtime)
            files = os.listdir(dir)
            root.title('Info reader / '+selected_folder+" / "+str(len(files))+" files")
            # files.sort(key=os.path.getctime)
            files.sort(key=lambda f: os.path.getmtime(os.path.join(dir,f)), reverse=True)
            # files = files.sort(key=lambda f: os.path.getmtime(os.path.join(dir, f)))
            # files_path_name["============"+dir+"============"] = ""
            for f in files:
                if not any(format_img in f for format_img in img_formats):continue
                path = os.path.join(dir,f)
                try:
                    if textbox2.get("1.0",END)[:-1] == "" or textbox2.get("1.0",END)[:-1] in path:
                        files_path_name[f] = path
                except Exception as e:
                    files_path_name[dir+"/"+f] = path
                # files_path_name.append((path,f))
    return files_path_name
files_path_name = refresh_files(current_subdirs)

def resize_image_for_panel(img):
    x, y = img.size
    if x > max_size or y > max_size:
        if x > y:
            y = int(1/x*max_size*y)
            x = max_size
        else:
            x = int(1/y*max_size*x)
            y = max_size
    return img.resize((x,y), Image.Resampling.LANCZOS)

def display_image_on_panel(selected_langs):
    files_path = files_path_name[selected_langs]
    img = Image.open(files_path)
    img = resize_image_for_panel(img)
    img = ImageTk.PhotoImage(img)
    panel.configure(image = img)
    panel.image = img

def get_model_name_from_infos(infos):
    infos=infos.split("\n")
    if len(infos) > 0:
        infos=infos[len(infos)-1].split(", ")
        for i in infos:
            if "Model hash" in i:
                model_hash = i.split(" ")[2]
                m = [e for e in models if model_hash in e]
                if len(m) > 0:
                    return m[0]
    return ""

def display_infos_on_text(selected_langs):
    infos = get_infos(files_path_name[selected_langs])
    if infos != "No description." and infos != "":
        m     = get_model_name_from_infos(infos)
        if m != "":
            infos = infos+"\n\n"+m
    textbox.delete("1.0", "end")
    textbox.insert(END,infos)
    change_text_color(textbox, "Negative prompt:", "#ffa500")

def get_path_from_selected_image():
    selected_indices = listbox.curselection()
    selected_langs = ",".join([listbox.get(i) for i in selected_indices])
    return selected_langs

def get_folder_from_folder_listbox():
    selected_indices = listbox_folders.curselection()
    selected_folder = ",".join([listbox_folders.get(i) for i in selected_indices])
    return selected_folder

def items_selected(event=""):
    selected_langs = get_path_from_selected_image()
    try:
        if files_path_name[selected_langs] != "" and files_path_name[selected_langs] != None and ".png" in files_path_name[selected_langs]:
            display_image_on_panel(selected_langs)
            display_infos_on_text(selected_langs)
        # elif os.isdir(os.join(current_subdirs,selected_langs)):
        #     print("youpi")
        # elif os.path.isdir(files_path_name[selected_langs]):
        #     print(files_path_name[selected_langs])
    except Exception as e:
        pass

def folder_selected(event=""):
    global selected_folder
    selected_folder_temp = get_folder_from_folder_listbox()
    # if selected_folder_temp == "Up":
    #     selected_folder = "."
    #     daemonizer(refresh_folders,os.path.join(selected_folder))
    if selected_folder_temp != "":
        same_folder = selected_folder == selected_folder_temp
        selected_folder = selected_folder_temp
        daemonizer(refresh_stringvar_f5)
        # daemonizer(refresh_folders,os.path.join(current_dir,selected_folder))
        print(os.path.join(current_dir,selected_folder))
        # if not same_folder:
        #     daemonizer(refresh_folders,os.path.join(selected_folder))


def upKey(event):
    change_list_selection(-1)
def downKey(event):
    change_list_selection(1)

def change_list_selection(key):
    selected_indices = listbox.curselection()
    listbox.selection_clear(0,END)
    listbox.selection_set(selected_indices[0]+key)
    items_selected()

def refresh_stringvar_f5(event=""):
    # while True:
    global current_subdirs
    current_subdirs = [ name for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name)) ]
    refresh_files(current_subdirs)
    refresh_stringvar(files_path_name)
    # sleep(3)

def open_image_from_panel(path):
    files_path = files_path_name[path]
    img = Image.open(files_path)
    # imgglass = "C:\Program Files\ImageGlass\ImageGlass.exe".replace(' ', '\\ ')
    # command = imgglass+" "+files_path
    # os.system(command)
    img.show()

def show_image(event):
    selected_langs = get_path_from_selected_image()
    if files_path_name[selected_langs] != "":
        open_image_from_panel(selected_langs)

def copy_to_clipboard():
    infos = get_infos(files_path_name[get_path_from_selected_image()])
    pyperclip.copy(infos)

def copy_to_clipboard_no_underscore():
    infos = get_infos(files_path_name[get_path_from_selected_image()])
    infos = infos.replace("_"," ")
    pyperclip.copy(infos)

def clean_clipboard():
    infos = pyperclip.paste()
    infos = infos.replace("_"," ")
    pyperclip.copy(infos)

def refresh_folders(path=selected_folder):
    path="." # TO REMOVE
    if path == "Up":return
    folders = [join(path,f) for f in os.listdir(path) if os.path.isdir(join(path,f))]
    folders.sort(reverse=True)
    listbox_folders.delete(0, 'end')
    for folder in folders:
        listbox_folders.insert(0, folder[2:])
    # listbox_folders.insert(0, "Up")

# def refresh_folders():
#     # listbox_folders.delete(0, 'end')
#     folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
#     # for folder in folders:
#     #     w = os.walk(folder)
#     #     for ww in w:
#     #         if os.path.isdir(os.path.join(current_dir,ww)):
#     #             print(ww)
#         # listbox_folders.insert(0, folder)
#     # folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
#     folders.sort(reverse=True)
#     listbox_folders.delete(0, 'end')
#     for folder in folders:
#         listbox_folders.insert(0, folder)
        # subfolders = [x[0] for x in os.walk(folder)]
        # for subfolder in subfolders:
        #     print(subfolder)
            # listbox_folders.insert(0, subfolder)
    # for folder in folders:
    #     listbox_folders.insert(0, str(os.path.join(path, folder))[2:])

def on_key_press(event):
    daemonizer(refresh_stringvar_f5)

def change_text_color(text_widget, text_to_change, color):
    start = text_widget.search(text_to_change, '1.0', stopindex='end')
    end = f"{start}+{len(text_to_change)}c"
    text_widget.tag_add("color", start, end)
    text_widget.tag_config("color", foreground=color)
    text_widget.tag_config("color", foreground=color, font=(font_used, font_size, "bold"))

daemonizer(refresh_stringvar_f5)
root = Tk()
root.title('Info reader')
root.geometry("1560x800")
file_names = StringVar()
bg_color="#1c1c1c"
fg_color="#d9d9d9"
hl_color="#0000d9"
font_size=10
font_used="Roboto"
border_size = 3
root.config(bg=bg_color)
textbox = Text(root,wrap=WORD, bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
textbox2 = Text(root,wrap=WORD, bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
panel = Label(root,image=None, bg=bg_color, fg=fg_color, bd=border_size, highlightcolor=hl_color, highlightbackground=hl_color)
listbox = Listbox(root, listvariable=file_names, height=10, selectmode="SINGLE", bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
listbox_folders = Listbox(root, height=10, selectmode="SINGLE", bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
refresh_stringvar(files_path_name)
button = Button(root,text="Copy",command=copy_to_clipboard, bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
button2 = Button(root,text="Copy no _",command=copy_to_clipboard_no_underscore, bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)
button3 = Button(root,text="Clean _",command=clean_clipboard, bg=bg_color, fg=fg_color, bd=border_size, font=(font_used, font_size), highlightcolor=hl_color, highlightbackground=hl_color)

listbox.place(x=max_size+180,y=10,width=840, height=500)

scrollbar = Scrollbar(root, highlightcolor=hl_color, highlightbackground=hl_color)
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)
scrollbar.place(x=max_size+180+840, y=10, height=500)

listbox_folders.place(x=max_size+20,y=10,width=160, height=500)
panel.place(x=10,y=10,width=max_size)
textbox.place(x=10,y=540,width=1460,height=240)
textbox2.place(x=max_size+20,y=512,width=1018,height=26) #searchbox
button.place(x=1475,y=540,width=74,height=50)
button2.place(x=1475,y=600,width=74,height=50)
button3.place(x=1475,y=660,width=74,height=50)

textbox2.bind("<KeyRelease>", on_key_press)
panel.bind('<Double-Button-1>', show_image)
listbox.bind('<<ListboxSelect>>', items_selected)
listbox_folders.bind('<<ListboxSelect>>', folder_selected)
refresh_folders()

root.bind('<Up>', upKey)
root.bind('<Down>', downKey)
root.bind('<Escape>', exit)
root.bind('<F5>', refresh_stringvar_f5)

root.mainloop()

exit()
