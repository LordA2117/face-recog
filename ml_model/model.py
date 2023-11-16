import face_recognition
import os
from PIL import Image, ImageDraw


sample_images = "./sample_images"

known_faces = os.listdir(sample_images)
known_faces_encodings = []
known_faces_names = [i.split(".")[0] for i in known_faces]

for file in known_faces:
    image = face_recognition.load_image_file(f"{sample_images}/{file}")
    known_faces_encodings.append(face_recognition.face_encodings(image)[0])

unknown_image = input("Enter the name of the unknown image: ")
unknown_image = face_recognition.load_image_file(unknown_image)
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image) # Drawing the image we created from the image array
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]


results = face_recognition.compare_faces(known_faces_encodings, unknown_image_encoding)
face_locations = face_recognition.face_locations(unknown_image) # Locations of all the faces in the image (every single one)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Drawing faces only for a matching face
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    face_name = None
    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
    if True in matches:
        face_name = known_faces_names[matches.index(True)]
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        #print("Found bro in", matches)
    else:
        #print("Did not find keenu in", matches)
        pass

del draw

#Display the obtained image
pil_image.show()

#num_matched_faces = results.count(True)
#print(f"Found {num_matched_faces} matched faces in the image")
#print(f"The face is of {known_faces_names[results.index(True)]}")
#print(results)

# TODO: Find out exactly what each variable is in the for loop, and the make the program draw a box only around our target's face
