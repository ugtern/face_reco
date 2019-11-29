import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

known_image = face_recognition.load_image_file("./images/110635693_5ddafd6d278bc.jpg")
unknown_image = face_recognition.load_image_file("./images/110635693_5ddafd6d06179.jpg")

pil_im = Image.open('./images/110635693_5ddafd6d278bc.jpg')
pil_im.show()

client_encoding = face_recognition.face_encodings(known_image)[0]

face_location = face_recognition.face_locations(unknown_image)
face_encoding = face_recognition.face_encodings(unknown_image, face_location)

print(face_location)
print(face_encoding)

known_face_encodings = [
    client_encoding
]
known_face_names = [
    "Client"
]

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
for (top, right, bottom, left), face_encoding in zip(face_location, face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = 'Unknown'

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print(face_distances)
    best_match_index = np.argmin(face_distances)
    print('best_match_index', best_match_index)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

pil_image.save('result.png', 'PNG')

results = face_recognition.compare_faces([client_encoding], face_encoding)
print(results)
