import tkinter as tk
from tkinter import messagebox, filedialog
from utils_en.predict import load_image
from PIL import Image, ImageTk


def GUI():

    def click_submit():
        image_path = click_submit.image_path
        # Load the image path into image_path

        result = load_image(image_path)
        # Put the path into the function to make a prediction

        messagebox.showinfo("Result:", result)
        # Display the result in a popup window

    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', "*.jpg;*.png")])
        # Open a file selection dialog and let the user choose a file, only show .jpg and .png files

        if file_path:
            click_submit.image_path = file_path
            # If a file is selected, save the path for the click_submit function to use

            img = Image.open(file_path)
            # Load the image from the selected file path into img

            img = img.resize((227, 227))
            # Set the display size

            img = ImageTk.PhotoImage(img)
            # Convert the PIL image to a PhotoImage object usable by tkinter

            image_label.config(image=img)
            # Update the image of the image_label to img

            image_label.image = img
            # Keep a reference to the image until it is replaced next time

    root = tk.Tk()
    root.title('Sports Recognition')
    root.geometry('400x380+500+250')
    # Define the main window title, size, and distance from the top-left corner

    upload_button = tk.Button(root, text='Upload Image', width=13, height=3, command=upload_image)
    upload_button.pack(pady=10)
    # Create a button that calls the upload_image function when clicked
    # Set the display text, size, and vertical margin of the button

    image_label = tk.Label(root)
    image_label.pack()
    # Define a label for displaying the image

    click_submit = tk.Button(root, text='Identify', width=7, height=2, command=click_submit)
    click_submit.pack(pady=10)
    # Create a button that calls the click_submit function when clicked

    root.mainloop()
    # Start the Tkinter main event loop, this window starts running and waits for interaction
