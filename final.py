import tkinter as tk
from tkinter import filedialog
import ocr
from predict import test_model
# from predict_2 import test_model_1


# Main Application Class
class ModelApp(tk.Tk):

    # Initializing the Application
    def __init__(self):
        super().__init__()

        self.title("Model Options")
        self.geometry("400x300")

        self.option_var = tk.StringVar()
        self.user_input = ""
        self.selected_image_path = ""
        self.result_text = ""

        self.create_option_page()

    # Creating Option Page
    def create_option_page(self):
        self.clear_frame()  # Clear the current widgets on the frame

        label = tk.Label(self, text="Select your Converting choice:")
        label.pack(pady=10)

        options = [
            "character recognition",
            # "my_dataset",
            "OCR recognition"
        ]

        option_menu = tk.OptionMenu(self, self.option_var, *options)
        option_menu.pack(pady=5)
        option_menu.config(bg="#CDC8B1")

        next_button = tk.Button(self, text="Submit" , command=self.create_image_page)
        next_button.pack(pady=10)
        next_button.config(text="submit!", bg="yellow")

    # Creating Image Page
    def create_image_page(self):
        self.clear_frame() # Clear the current widgets on the frame

        label = tk.Label(self, text=" Converting Marathi Inscript to Marathi Text")
        label.pack(pady=10)

        upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        upload_button.pack()


        next_button = tk.Button(self, text="submit!",command=self.perform_selected_option)
        next_button.pack(pady=10)
        next_button.config(bg="#76EEC6")


        back_button = tk.Button(self,text='back', command=self.create_option_page)
        back_button.pack(pady=5)
        back_button.config(bg="#CDB79E")

    # Performing Selected Option
    def perform_selected_option(self):
        selected_option = self.option_var.get()

        if selected_option == "character recognition":
            result = test_model(self.selected_image_path)
        # elif selected_option == "my_dataset":
        #     result = test_model_1(self.selected_image_path)
        elif selected_option == "OCR recognition":
            result = ocr.perform_ocr(self.selected_image_path)
        else:
            result = "Invalid option selected."

        result_window = tk.Toplevel(self)
        result_window.title("Option Result")

        text_label = tk.Label(result_window, text=result, wraplength=500)
        text_label.pack(pady=50)

    # This method is used to clear the current pages widgets, preparing the frame for the next page
    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

    # Uploading Image
    def upload_image(self):
        self.selected_image_path = filedialog.askopenfilename()


# This block checks if the script is being run directly (not imported as a module). If so, it creates an instance
# of the ModelApp class and starts the main event loop to run the application.
if __name__ == "__main__":
    app = ModelApp()
    app.mainloop()
