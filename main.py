import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

known_image = face_recognition.load_image_file("./images/IMG_0441.jpg") # в эту переменную кладем картинку с лицом
known_image2 = face_recognition.load_image_file("./images/3.jpg")
known_image3 = face_recognition.load_image_file("./images/4.jpg")
unknown_image = face_recognition.load_image_file("./images/IMG_4005.jpg") # сюда кладем ту фотку на которой надо распознать лица

# pil_im = Image.open('./images/2.jpg')
# pil_im.show()

client_encoding = face_recognition.face_encodings(known_image)[0]
client2_encoding = face_recognition.face_encodings(known_image2)[0]
client3_encoding = face_recognition.face_encodings(known_image3)[0]

face_location = face_recognition.face_locations(unknown_image)
face_encoding = face_recognition.face_encodings(unknown_image, face_location)

print(face_location)
print(face_encoding)

known_face_encodings = [
    client_encoding,
    client3_encoding,
    client2_encoding
] # переменные с лицами
known_face_names = [
    "skruj",
    "Borj",
    "Juran"
] # имена для лиц

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
i = 0
for (top, right, bottom, left), face_encoding in zip(face_location, face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = 'Unknown'
    i += 1

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print('face_distances')
    print(face_distances)
    best_match_index = np.argmin(face_distances)
    print('best_match_index', best_match_index)
    print(known_face_names[best_match_index])

    name = known_face_names[best_match_index] # выбор имени для отрисовки на картинке

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    print(str(face_distances[0]))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    # pil_image.save('result' + str(i) + '.png', 'PNG')

pil_image.save('result.png', 'PNG')

results = face_recognition.compare_faces([client_encoding], face_encoding)
print(results)
