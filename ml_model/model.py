from termcolor import colored
from alive_progress import alive_bar
print(colored("[*] Importing libraries", "cyan"))

import face_recognition
import os
from PIL import Image, ImageDraw

os.system("clear")

sample_images = "./sample_images"

print(colored("[*] Loading database", "cyan"))

known_faces = os.listdir(sample_images)
known_faces_encodings = []
known_faces_names = [i.split(".")[0] for i in known_faces]
num_known = len(known_faces)

print(colored("[*] Processing Sample Images. This process may take some time....", "cyan"))

with alive_bar(num_known) as bar:
    for file in known_faces:
        image = face_recognition.load_image_file(f"{sample_images}/{file}")
        known_faces_encodings.append(face_recognition.face_encodings(image)[0])
        bar()

check_mark = u"\u2713"
print(colored("[#] Available sample images:", "yellow"))
for i in os.listdir("./"):
    if ".jpg" in i or ".jpeg" in i or ".png" in i:
        print(colored(f"[{check_mark}] {i}", "green"))

unknown_image = input("Enter the name of the unknown image: ")
unknown_image = face_recognition.load_image_file(unknown_image)
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image) # Drawing the image we created from the image array
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]


results = face_recognition.compare_faces(known_faces_encodings, unknown_image_encoding)
face_locations = face_recognition.face_locations(unknown_image) # Locations of all the faces in the image (every single one)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
num_execs = len(face_locations)
# Drawing faces only for a matching face
with alive_bar(num_execs) as bar:
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        bar()
        face_name = None
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        if True in matches:
            #print(colored("[+] Faces from the sample images have been detected in the given image.", "green"))
            face_name = known_faces_names[matches.index(True)]
            draw.rectangle(((left, top), (right, bottom)), outline=(255, 34, 38), width=5)
        else:
            pass

del draw

print(colored("[+] Displaying the image...", "green"))

#Display the obtained image
pil_image.show()

# Logging:
#num_matched_faces = results.count(True)
#print(f"Found {num_matched_faces} matched faces in the image")
#print(f"The face is of {known_faces_names[results.index(True)]}")
#print(results)

