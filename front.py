from tkinter import *
from final import get_response_for_gui

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.bot = "mybot"

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("****chat with me!****")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="WELCOME!", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        self.text_Widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_Widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_Widget.configure(cursor="arrow", state=DISABLED)
        
        scrollbar = Scrollbar(self.text_Widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_Widget.yview)
        
        BOTTOM_label = Label(self.window, bg=BG_GRAY, height=80)
        BOTTOM_label.place(relwidth=1, rely=0.825)
        
        self.msg_entry = Entry(BOTTOM_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        send_button = Button(BOTTOM_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "you")

    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_Widget.configure(state=NORMAL)
        self.text_Widget.insert(END, msg1)
        self.text_Widget.configure(state=DISABLED)
        msg2 = f"{self.bot}: {get_response_for_gui(msg)}\n\n"
        self.text_Widget.configure(state=NORMAL)
        self.text_Widget.insert(END, msg2)
        self.text_Widget.configure(state=DISABLED)
        self.text_Widget.see(END)

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
